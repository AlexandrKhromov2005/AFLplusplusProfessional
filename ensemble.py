#!/usr/bin/env python3
"""
ensemble.py — AFL++ Ensemble vs Baseline Orchestrator.
Одна команда: запускает 6 фаззеров (3 ML + 3 baseline), TUI дашборд, CSV + PNG на выходе.
"""

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import yaml


@dataclass
class InstanceConfig:
    name: str
    binary: str           # "mlf-fuzz" или "afl-fuzz"
    role: str             # "main" или "secondary"
    group: str            # "ml" или "baseline"
    sync_dir: str         # путь к sync_dir этой группы
    input_dir: str        # путь к input корпусу этой группы
    env: dict = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    target: str
    input_ml: str
    input_baseline: str
    output_ml: str
    output_baseline: str
    dict_path: Optional[str]
    timeout_ms: int
    poll_interval_sec: int
    common_env: dict
    instances: List[InstanceConfig]  # Все 6 инстансов


@dataclass
class InstanceStats:
    name: str
    group: str = ""
    alive: bool = False
    paths_total: int = 0
    paths_found: int = 0
    edges_found: int = 0
    bitmap_cvg: float = 0.0
    execs_per_sec: float = 0.0
    last_find_time: int = 0
    start_time: int = 0
    cycles_done: int = 0
    pending_total: int = 0
    pending_favs: int = 0
    stag_mode: str = "—"
    revisited: int = 0


# ── ANSI цвета ──

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GRAY    = "\033[0;90m"
    WHITE   = "\033[0;37m"
    BWHITE  = "\033[1;37m"
    RED     = "\033[1;31m"
    GREEN   = "\033[1;32m"
    YELLOW  = "\033[0;33m"
    BYELLOW = "\033[1;33m"
    BLUE    = "\033[1;34m"
    MAGENTA = "\033[1;35m"
    CYAN    = "\033[1;36m"
    BCYAN   = "\033[1;36m"


# ── Парсинг конфига ──

def load_config(path: str) -> EnsembleConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    instances = []

    output_ml = os.path.expanduser(raw['output_ml'])
    output_bl = os.path.expanduser(raw['output_baseline'])

    # Раздельные input корпуса, с fallback на старое поле 'input'
    input_ml = os.path.expanduser(raw.get('input_ml', raw.get('input', '')))
    input_bl = os.path.expanduser(raw.get('input_baseline', raw.get('input', '')))

    for inst in raw.get('ml_instances', []):
        instances.append(InstanceConfig(
            name=inst['name'],
            binary=inst['binary'],
            role=inst['role'],
            group='ml',
            sync_dir=output_ml,
            input_dir=input_ml,
            env=inst.get('env', {}),
        ))

    for inst in raw.get('baseline_instances', []):
        instances.append(InstanceConfig(
            name=inst['name'],
            binary=inst['binary'],
            role=inst['role'],
            group='baseline',
            sync_dir=output_bl,
            input_dir=input_bl,
            env=inst.get('env', {}),
        ))

    return EnsembleConfig(
        target=os.path.expanduser(raw['target']),
        input_ml=input_ml,
        input_baseline=input_bl,
        output_ml=output_ml,
        output_baseline=output_bl,
        dict_path=os.path.expanduser(raw['dict']) if raw.get('dict') else None,
        timeout_ms=raw.get('timeout_ms', 10000),
        poll_interval_sec=raw.get('poll_interval_sec', 5),
        common_env=raw.get('common_env', {}),
        instances=instances,
    )


# ── Запуск инстансов и демонов ──

def _find_binary(name: str) -> str:
    """Найти бинарник по имени: сначала рядом, потом в PATH, потом поиском."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'AFLplusplus', name),
        os.path.join(script_dir, name),
    ]
    parent = os.path.dirname(script_dir)
    candidates += [
        os.path.join(parent, 'AFLplusplus', name),
        os.path.join(parent, name),
    ]
    home = os.path.expanduser('~')
    candidates += [
        os.path.join(home, 'AFLplusplusProfessional', 'AFLplusplus', name),
        os.path.join(home, 'AFLplusplus', name),
        os.path.join(home, 'projects', 'AFLplusplusProfessional', 'AFLplusplus', name),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Cannot find '{name}'. Searched:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + f"\n  PATH ({name} not in $PATH)"
    )


def _find_ml_daemon_dir() -> str:
    """Найти директорию содержащую ml_daemon/ пакет."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.dirname(script_dir),
        os.path.join(os.path.expanduser('~'), 'AFLplusplusProfessional'),
        os.path.join(os.path.expanduser('~'), 'projects', 'AFLplusplusProfessional'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'ml_daemon')):
            return c
    raise FileNotFoundError(
        "Cannot find ml_daemon/ package. Searched:\n"
        + "\n".join(f"  {c}/ml_daemon/" for c in candidates)
    )


def _find_ml_model() -> str:
    """Найти mlf_model_v1.pkl."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser('~')
    candidates = [
        os.path.join(script_dir, 'mlf_model_v1.pkl'),
        os.path.join(os.path.dirname(script_dir), 'mlf_model_v1.pkl'),
        os.path.join(home, 'AFLplusplusProfessional', 'mlf_model_v1.pkl'),
        os.path.join(home, 'projects', 'AFLplusplusProfessional', 'mlf_model_v1.pkl'),
        '/tmp/mlf_model.pkl',
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    # Model file is optional — daemon creates it on first training
    return '/tmp/mlf_model.pkl'


_binary_cache = {}

def resolve_binary(name: str) -> str:
    if name not in _binary_cache:
        _binary_cache[name] = _find_binary(name)
    return _binary_cache[name]


def build_cmd(config: EnsembleConfig, inst: InstanceConfig) -> list:
    binary = resolve_binary(inst.binary)
    cmd = [binary]
    cmd += ["-i", inst.input_dir]
    cmd += ["-o", inst.sync_dir]

    if inst.role == "main":
        cmd += ["-M", inst.name]
    else:
        cmd += ["-S", inst.name]

    cmd += ["-m", "none"]
    cmd += ["-t", str(config.timeout_ms)]

    if config.dict_path:
        cmd += ["-x", config.dict_path]

    cmd += ["--", config.target]
    return cmd


def build_env(config: EnsembleConfig, inst: InstanceConfig) -> dict:
    env = os.environ.copy()
    env.update(config.common_env)
    env.update(inst.env)
    return env


def launch_all(config: EnsembleConfig):
    """Запустить демоны + все 6 инстансов. Возвращает (procs, daemon_procs, stderr_handles)."""
    daemon_procs = []
    procs = {}
    stderr_handles = {}

    # ── Создать sync_dir (нужно до запуска — для stderr логов) ──
    os.makedirs(config.output_ml, exist_ok=True)
    os.makedirs(config.output_baseline, exist_ok=True)

    # ── Демоны для mlf-fuzz инстансов ──
    for inst in config.instances:
        if inst.binary == 'mlf-fuzz':
            log_path = inst.env.get('MLF_TRAINING_LOG', f'/tmp/mlf_training_{inst.name}.bin')
            socket_path = f'/tmp/mlf_scheduler_{inst.name}.sock'
            model_path = _find_ml_model()

            # Удалить старый сокет если остался
            if os.path.exists(socket_path):
                os.unlink(socket_path)

            inst.env['MLF_SCHEDULER_SOCKET'] = socket_path

            daemon_cmd = [
                sys.executable, '-m', 'ml_daemon.daemon',
                '--socket', socket_path,
                '--log', log_path,
                '--model', model_path,
            ]
            print(f"  Daemon for {inst.name}: socket={socket_path}")
            daemon_stderr_log = os.path.join(config.output_ml, f'daemon_{inst.name}.log')
            daemon_stderr_fh = open(daemon_stderr_log, 'w')
            dp = subprocess.Popen(
                daemon_cmd,
                cwd=_find_ml_daemon_dir(),
                stdout=subprocess.DEVNULL,
                stderr=daemon_stderr_fh,
                preexec_fn=os.setsid,
            )
            daemon_procs.append(dp)
            stderr_handles[f'daemon_{inst.name}'] = daemon_stderr_fh
            time.sleep(1.5)

    # ── Фаззеры ──
    # Сначала main, потом secondary (main должен создать sync_dir структуру)
    sorted_instances = sorted(config.instances, key=lambda i: (0 if i.role == 'main' else 1))

    for inst in sorted_instances:
        cmd = build_cmd(config, inst)
        env = build_env(config, inst)

        stderr_log = os.path.join(inst.sync_dir, f'{inst.name}_stderr.log')
        stderr_fh = open(stderr_log, 'w')

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
            preexec_fn=os.setsid,
        )
        procs[inst.name] = proc
        stderr_handles[inst.name] = stderr_fh
        tag = "ML" if inst.group == "ml" else "BL"
        role_flag = "-M" if inst.role == "main" else "-S"
        print(f"  [{tag}] {inst.name} (PID {proc.pid}): {inst.binary} {role_flag}")
        time.sleep(0.5)

    return procs, daemon_procs, stderr_handles


# ── Парсинг fuzzer_stats ──

def parse_fuzzer_stats(stats_path: str, name: str, group: str) -> InstanceStats:
    stats = InstanceStats(name=name, group=group)

    if not os.path.exists(stats_path):
        return stats

    try:
        with open(stats_path) as f:
            lines = f.readlines()
    except (IOError, OSError):
        return stats

    stats.alive = True
    kv = {}
    for line in lines:
        if ':' in line:
            key, _, val = line.partition(':')
            kv[key.strip()] = val.strip()

    stats.start_time     = int(kv.get('start_time', 0))
    stats.last_find_time = int(kv.get('last_find_time', 0))
    stats.cycles_done    = int(kv.get('cycles_done', 0))
    stats.paths_total    = int(kv.get('corpus_count', kv.get('paths_total', 0)))
    stats.paths_found    = int(kv.get('corpus_found', kv.get('paths_found', 0)))
    stats.pending_total  = int(kv.get('pending_total', 0))
    stats.pending_favs   = int(kv.get('pending_favs', 0))
    stats.edges_found    = int(kv.get('edges_found', 0))

    # bitmap_cvg: "0.24% / 26.15%" → берём второе число (total coverage)
    cvg_str = kv.get('bitmap_cvg', '0')
    if '/' in cvg_str:
        cvg_str = cvg_str.split('/')[1]
    cvg_str = cvg_str.replace('%', '').strip()
    try:
        stats.bitmap_cvg = float(cvg_str)
    except ValueError:
        stats.bitmap_cvg = 0.0

    speed_str = kv.get('execs_per_sec', '0')
    try:
        stats.execs_per_sec = float(speed_str)
    except ValueError:
        stats.execs_per_sec = 0.0

    return stats


def read_all_stats(config: EnsembleConfig, procs: dict) -> List[InstanceStats]:
    all_stats = []
    for inst in config.instances:
        stats_path = os.path.join(inst.sync_dir, inst.name, "default", "fuzzer_stats")
        if not os.path.exists(stats_path):
            stats_path = os.path.join(inst.sync_dir, inst.name, "fuzzer_stats")

        stats = parse_fuzzer_stats(stats_path, inst.name, inst.group)

        proc = procs.get(inst.name)
        if proc and proc.poll() is not None:
            stats.alive = False

        all_stats.append(stats)

    return all_stats


# ── Вспомогательные функции рендера ──

def fmt_num(n: int) -> str:
    return f"{n:,}"


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def fmt_time_ago(timestamp: int) -> str:
    if timestamp == 0:
        return "—"
    elapsed = max(0, int(time.time()) - timestamp)
    if elapsed < 60:
        return f"{elapsed}s"
    elif elapsed < 3600:
        m, s = divmod(elapsed, 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(elapsed, 3600)
        m = rem // 60
        return f"{h}h {m:02d}m"


def fmt_elapsed(start_time: int) -> str:
    elapsed = max(0, int(time.time()) - start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def color_last_find(timestamp: int) -> str:
    if timestamp == 0:
        return C.GRAY
    elapsed = int(time.time()) - timestamp
    if elapsed < 10:
        return C.GREEN
    elif elapsed > 300:
        return C.RED
    return C.WHITE


def color_instance(name: str) -> str:
    n = name.lower()
    if 'exploit' in n:   return C.MAGENTA
    if 'explore' in n:   return C.BLUE
    if 'havoc' in n:     return C.YELLOW
    if 'base' in n:      return C.WHITE
    return C.WHITE


def color_stag(stag: str) -> str:
    if stag in ("REVIS", "REVISIT"):
        return C.RED
    return C.GRAY


def color_delta(value: float) -> str:
    """Зелёный если ML впереди (>0), красный если отстаёт (<0)."""
    if value > 0:
        return C.GREEN
    elif value < 0:
        return C.RED
    return C.GRAY


def fmt_delta(val: float, is_pct: bool = False) -> str:
    """Форматировать дельту: +1,299 или +10.4%"""
    if is_pct:
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.1f}%"
    else:
        sign = "+" if val >= 0 else ""
        return f"{sign}{int(val):,}"


# ── Рендер таблицы ──

# Ширины колонок
#                inst   finds  edges  cov%   exec/s corpus stag   last_find
COL_W =         [10,    8,     8,     9,     8,     9,     7,     14]
NUM_COLS = len(COL_W)


def _hline(left: str, mid: str, right: str) -> str:
    """Горизонтальная линия из box-drawing символов."""
    segments = ['─' * w for w in COL_W]
    return f"{C.GRAY}{left}{mid.join(segments)}{right}{C.RESET}"


def _row(cells) -> str:
    """Строка таблицы.
    cells: list of (text, align, color) — len == NUM_COLS.
    text: строка (без ANSI!), будет паддиться.
    align: 'l', 'r', 'c'.
    color: ANSI color code.
    """
    parts = []
    for i, (text, align, color) in enumerate(cells):
        w = COL_W[i]
        # Обрезать если не влезает
        if len(text) > w:
            text = text[:w - 1] + '…'
        # Паддинг
        if align == 'r':
            padded = text.rjust(w)
        elif align == 'c':
            padded = text.center(w)
        else:
            padded = text.ljust(w)
        parts.append(f"{color}{padded}{C.RESET}")
    inner = f"{C.GRAY}│{C.RESET}".join(parts)
    return f"{C.GRAY}│{C.RESET}{inner}{C.GRAY}│{C.RESET}"


def _header_row_full_width(text: str, color: str) -> str:
    """Строка-заголовок, занимающая всю ширину таблицы."""
    inner_w = sum(COL_W) + (NUM_COLS - 1)  # ширина между крайними │
    padded = text.ljust(inner_w)
    if len(padded) > inner_w:
        padded = padded[:inner_w]
    return f"{C.GRAY}│{C.RESET}{color}{padded}{C.RESET}{C.GRAY}│{C.RESET}"


def _compute_group_totals(stats_list: List[InstanceStats]) -> dict:
    """Агрегаты по группе."""
    return {
        'finds':     sum(s.paths_found for s in stats_list),
        'edges':     max((s.edges_found for s in stats_list), default=0),
        'cov':       max((s.bitmap_cvg for s in stats_list), default=0.0),
        'exec_s':    sum(s.execs_per_sec for s in stats_list),
        'last_find': max((s.last_find_time for s in stats_list), default=0),
        'alive':     sum(1 for s in stats_list if s.alive),
    }


def render_table(all_stats: List[InstanceStats], config: EnsembleConfig,
                 ensemble_start: int):
    """Отрисовать полную TUI таблицу: ML group + Baseline group + comparison."""

    ml_stats = [s for s in all_stats if s.group == 'ml']
    bl_stats = [s for s in all_stats if s.group == 'baseline']
    ml_tot = _compute_group_totals(ml_stats)
    bl_tot = _compute_group_totals(bl_stats)

    target_name = os.path.basename(config.target)
    elapsed_str = fmt_elapsed(ensemble_start)
    alive_total = ml_tot['alive'] + bl_tot['alive']
    total_count = len(all_stats)

    lines = []

    # ── Clear screen ──
    lines.append("\033[H\033[J")

    # ── Top border ──
    lines.append(_hline('┌', '┬', '┐'))

    # ── Title ──
    title = f"  ENSEMBLE vs BASELINE  {elapsed_str}   target: {target_name}   {alive_total}/{total_count} alive"
    lines.append(_header_row_full_width(title, C.BCYAN))

    # ── Column headers ──
    lines.append(_hline('├', '┼', '┤'))
    lines.append(_row([
        (' instance', 'l', C.BWHITE),
        ('finds',     'r', C.BWHITE),
        ('edges',     'r', C.BWHITE),
        ('cov %',     'r', C.BWHITE),
        ('exec/s',    'r', C.BWHITE),
        ('corpus',    'r', C.BWHITE),
        ('stag',      'c', C.BWHITE),
        ('last find', 'r', C.BWHITE),
    ]))
    lines.append(_hline('├', '┼', '┤'))

    # ── Helper: instance row ──
    def inst_row(s: InstanceStats):
        if not s.alive:
            return _row([
                (f' {s.name}', 'l', C.GRAY),
                ('—', 'r', C.GRAY), ('—', 'r', C.GRAY), ('—', 'r', C.GRAY),
                ('—', 'r', C.GRAY), ('—', 'r', C.GRAY),
                ('DEAD', 'c', C.RED), ('—', 'r', C.GRAY),
            ])
        stag = s.stag_mode if s.stag_mode else '—'
        return _row([
            (f' {s.name}',                  'l', color_instance(s.name)),
            (fmt_num(s.paths_found),         'r', C.WHITE),
            (fmt_num(s.edges_found),         'r', C.WHITE),
            (fmt_pct(s.bitmap_cvg),          'r', C.WHITE),
            (fmt_num(int(s.execs_per_sec)),  'r', C.WHITE),
            (fmt_num(s.paths_total),         'r', C.WHITE),
            (stag,                           'c', color_stag(stag)),
            (fmt_time_ago(s.last_find_time), 'r', color_last_find(s.last_find_time)),
        ])

    # ── Helper: totals row ──
    def totals_row(label: str, tot: dict, color: str):
        return _row([
            (f' {label}',                   'l', color),
            (fmt_num(tot['finds']),          'r', color),
            (fmt_num(tot['edges']),          'r', color),
            (fmt_pct(tot['cov']),            'r', color),
            (fmt_num(int(tot['exec_s'])),    'r', color),
            ('—',                            'r', C.GRAY),
            ('',                             'c', C.GRAY),
            (fmt_time_ago(tot['last_find']), 'r', color_last_find(tot['last_find'])),
        ])

    # ── ML group ──
    for s in ml_stats:
        lines.append(inst_row(s))

    # ML TOTAL
    lines.append(totals_row('ML TOTAL', ml_tot, C.BYELLOW))

    # ── Separator between groups ──
    lines.append(_hline('├', '┼', '┤'))

    # ── Baseline group ──
    for s in bl_stats:
        lines.append(inst_row(s))

    # BL TOTAL
    lines.append(totals_row('BL TOTAL', bl_tot, C.BYELLOW))

    # ── Comparison separator ──
    lines.append(_hline('├', '┼', '┤'))

    # ── ML vs BL: абсолютная дельта ──
    d_finds = ml_tot['finds'] - bl_tot['finds']
    d_edges = ml_tot['edges'] - bl_tot['edges']
    d_cov   = ml_tot['cov']   - bl_tot['cov']
    d_exec  = ml_tot['exec_s'] - bl_tot['exec_s']

    lines.append(_row([
        (' ML vs BL',            'l', C.BWHITE),
        (fmt_delta(d_finds),     'r', color_delta(d_finds)),
        (fmt_delta(d_edges),     'r', color_delta(d_edges)),
        (fmt_delta(d_cov) + 'p', 'r', color_delta(d_cov)),
        (fmt_delta(d_exec),      'r', color_delta(d_exec)),
        ('',                     'r', C.GRAY),
        ('',                     'c', C.GRAY),
        ('',                     'r', C.GRAY),
    ]))

    # ── ML vs BL: процентная дельта ──
    def pct_delta(ml_val, bl_val):
        if bl_val == 0:
            return 0.0
        return ((ml_val - bl_val) / bl_val) * 100.0

    p_finds = pct_delta(ml_tot['finds'], bl_tot['finds'])
    p_edges = pct_delta(ml_tot['edges'], bl_tot['edges'])
    p_exec  = pct_delta(ml_tot['exec_s'], bl_tot['exec_s'])

    lines.append(_row([
        ('',                          'l', C.GRAY),
        (fmt_delta(p_finds, True),    'r', color_delta(p_finds)),
        (fmt_delta(p_edges, True),    'r', color_delta(p_edges)),
        ('',                          'r', C.GRAY),
        (fmt_delta(p_exec, True),     'r', color_delta(p_exec)),
        ('',                          'r', C.GRAY),
        ('',                          'c', C.GRAY),
        ('',                          'r', C.GRAY),
    ]))

    # ── Bottom border ──
    lines.append(_hline('└', '┴', '┘'))

    # ── Footer ──
    lines.append(f"  {C.GRAY}Ctrl+C → graceful stop, save CSV + PNG{C.RESET}")
    lines.append("")

    sys.stdout.write('\n'.join(lines))
    sys.stdout.flush()


# ── CSV Logger ──

class CSVLogger:
    def __init__(self, path: str, instance_names: list):
        self.path = path
        self.names = instance_names
        self.file = open(path, 'w', newline='')
        self.writer = csv.writer(self.file)

        header = ['timestamp', 'elapsed_sec']
        for name in instance_names:
            header += [f'{name}_finds', f'{name}_edges', f'{name}_cov',
                      f'{name}_exec_s', f'{name}_corpus']
        header += ['ml_total_finds', 'ml_total_edges', 'ml_total_cov',
                   'bl_total_finds', 'bl_total_edges', 'bl_total_cov',
                   'delta_finds', 'delta_edges', 'delta_cov']
        self.writer.writerow(header)

    def log(self, all_stats: List[InstanceStats], ensemble_start: int):
        now = int(time.time())
        elapsed = now - ensemble_start
        row_data = [now, elapsed]

        for s in all_stats:
            row_data += [s.paths_found, s.edges_found, s.bitmap_cvg,
                        int(s.execs_per_sec), s.paths_total]

        ml = [s for s in all_stats if s.group == 'ml']
        bl = [s for s in all_stats if s.group == 'baseline']

        ml_f = sum(s.paths_found for s in ml)
        ml_e = max((s.edges_found for s in ml), default=0)
        ml_c = max((s.bitmap_cvg for s in ml), default=0.0)
        bl_f = sum(s.paths_found for s in bl)
        bl_e = max((s.edges_found for s in bl), default=0)
        bl_c = max((s.bitmap_cvg for s in bl), default=0.0)

        row_data += [ml_f, ml_e, ml_c, bl_f, bl_e, bl_c,
                    ml_f - bl_f, ml_e - bl_e, ml_c - bl_c]
        self.writer.writerow(row_data)
        self.file.flush()

    def close(self):
        self.file.close()


# ── PNG график ──

def save_plot(csv_path: str, png_path: str, instance_names: list):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping PNG")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if len(rows) < 2:
        print("  Not enough data for plot")
        return

    minutes = [int(r['elapsed_sec']) / 60.0 for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('AFL++ Ensemble (ML) vs Baseline', fontsize=14, fontweight='bold')

    # ── Plot 1: Edges over time ──
    ax = axes[0]
    for name in instance_names:
        key = f'{name}_edges'
        if key in rows[0]:
            vals = [int(r[key]) for r in rows]
            group = 'ml' if 'base' not in name else 'baseline'
            style = '-' if group == 'ml' else '--'
            ax.plot(minutes, vals, style, label=name, alpha=0.7)
    ml_edges = [int(r['ml_total_edges']) for r in rows]
    bl_edges = [int(r['bl_total_edges']) for r in rows]
    ax.plot(minutes, ml_edges, '-', color='red', linewidth=2.5, label='ML TOTAL')
    ax.plot(minutes, bl_edges, '--', color='blue', linewidth=2.5, label='BL TOTAL')
    ax.set_ylabel('Edges (unique)')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Finds over time ──
    ax = axes[1]
    for name in instance_names:
        key = f'{name}_finds'
        if key in rows[0]:
            vals = [int(r[key]) for r in rows]
            group = 'ml' if 'base' not in name else 'baseline'
            style = '-' if group == 'ml' else '--'
            ax.plot(minutes, vals, style, label=name, alpha=0.7)
    ml_finds = [int(r['ml_total_finds']) for r in rows]
    bl_finds = [int(r['bl_total_finds']) for r in rows]
    ax.plot(minutes, ml_finds, '-', color='red', linewidth=2.5, label='ML TOTAL')
    ax.plot(minutes, bl_finds, '--', color='blue', linewidth=2.5, label='BL TOTAL')
    ax.set_ylabel('Own Finds')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Delta (ML - BL) over time ──
    ax = axes[2]
    d_edges = [int(r['delta_edges']) for r in rows]
    d_finds = [int(r['delta_finds']) for r in rows]
    ax.plot(minutes, d_edges, '-', color='red', linewidth=2, label='Δ edges')
    ax.plot(minutes, d_finds, '-', color='orange', linewidth=2, label='Δ finds')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('ML − Baseline')
    ax.set_xlabel('Time (minutes)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.fill_between(minutes, d_edges, 0,
                    where=[d >= 0 for d in d_edges], alpha=0.1, color='green')
    ax.fill_between(minutes, d_edges, 0,
                    where=[d < 0 for d in d_edges], alpha=0.1, color='red')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {png_path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='AFL++ Ensemble vs Baseline')
    parser.add_argument('--config', default='ensemble_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  AFL++ Ensemble vs Baseline Orchestrator")
    print(f"  Target:   {os.path.basename(config.target)}")
    print(f"  ML input: {config.input_ml}")
    print(f"  BL input: {config.input_baseline}")
    print(f"  ML sync:  {config.output_ml}")
    print(f"  BL sync:  {config.output_baseline}")
    print(f"  Instances: {len(config.instances)} ({sum(1 for i in config.instances if i.group=='ml')} ML + {sum(1 for i in config.instances if i.group=='baseline')} BL)")
    print(f"{'='*60}\n")

    # Auto-discover и показать пути
    print(f"  Resolving binaries...")
    for inst in config.instances:
        try:
            path = resolve_binary(inst.binary)
            print(f"    {inst.binary}: {path}")
        except FileNotFoundError as e:
            print(f"    {C.RED}ERROR: {e}{C.RESET}")
            sys.exit(1)

    try:
        daemon_dir = _find_ml_daemon_dir()
        print(f"    ml_daemon: {daemon_dir}/ml_daemon/")
    except FileNotFoundError as e:
        print(f"    {C.RED}ERROR: {e}{C.RESET}")
        sys.exit(1)

    try:
        model = _find_ml_model()
        print(f"    ML model: {model}")
    except FileNotFoundError as e:
        print(f"    {C.RED}ERROR: {e}{C.RESET}")
        sys.exit(1)

    print()

    procs, daemon_procs, stderr_handles = launch_all(config)

    ensemble_start = int(time.time())
    instance_names = [inst.name for inst in config.instances]

    # CSV
    csv_dir = config.output_ml
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'ensemble_log.csv')
    png_path = os.path.join(csv_dir, 'ensemble_plot.png')
    logger = CSVLogger(csv_path, instance_names)

    # Graceful shutdown
    shutdown = False

    def handle_sigint(sig, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\n  Waiting for fuzzer_stats...\n")
    time.sleep(5)

    try:
        while not shutdown:
            all_stats = read_all_stats(config, procs)
            render_table(all_stats, config, ensemble_start)
            logger.log(all_stats, ensemble_start)

            # Показать ошибки мёртвых инстансов
            for inst in config.instances:
                proc = procs.get(inst.name)
                if proc and proc.poll() is not None:
                    stderr_log = os.path.join(inst.sync_dir, f'{inst.name}_stderr.log')
                    if os.path.exists(stderr_log) and os.path.getsize(stderr_log) > 0:
                        with open(stderr_log) as f:
                            lines = f.readlines()[-5:]
                        if lines:
                            print(f"\n  {C.RED}[DEAD: {inst.name}]{C.RESET}")
                            for line in lines:
                                print(f"    {C.GRAY}{line.rstrip()}{C.RESET}")

            all_dead = all(not s.alive for s in all_stats)
            if all_dead and int(time.time()) - ensemble_start > 15:
                print("\n  All instances died. Exiting.")
                break

            for _ in range(config.poll_interval_sec * 10):
                if shutdown:
                    break
                time.sleep(0.1)

    finally:
        print("\n\n  Stopping all instances...")

        for name, proc in procs.items():
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                    print(f"    Stopped {name}")
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    print(f"    Killed {name}")

        for dp in daemon_procs:
            if dp.poll() is None:
                try:
                    os.killpg(os.getpgid(dp.pid), signal.SIGTERM)
                    dp.wait(timeout=3)
                except Exception:
                    try:
                        dp.kill()
                    except Exception:
                        pass

        for fh in stderr_handles.values():
            try:
                fh.close()
            except Exception:
                pass

        logger.close()
        print(f"\n  CSV:  {csv_path}")
        save_plot(csv_path, png_path, instance_names)

        elapsed = fmt_elapsed(ensemble_start)
        print(f"\n  Output ML:       {config.output_ml}")
        print(f"  Output Baseline: {config.output_baseline}")
        print(f"  Run time:        {elapsed}")
        print()


if __name__ == '__main__':
    main()
