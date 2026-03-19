"""
Парсер бинарного training log от mlf-fuzz.
Формат записи определён в include/ml-scheduler.h (ml_training_record_t).
"""

import struct
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Формат одной записи:
# 3×uint32 (entry_id, energy_assigned, new_edges)
# 1×uint32 (new_hit_counts) — итого 4×uint32 = 16 байт
# 1×uint8 (found_crash)
# 1×uint8 (found_hang)
# 6 байт паддинг (выравнивание)
# 16×float32 (features) = 64 байта
# 1×uint64 (timestamp_us) = 8 байт
RECORD_FORMAT = '4I2B6x16fQ'
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)
FEATURE_DIM = 16

FEATURE_NAMES = [
    'exec_us_norm', 'bitmap_size_norm', 'fuzz_level_norm', 'depth_norm',
    'handicap_norm', 'n_fuzz_norm', 'favored', 'passed_det',
    'is_ascii', 'tc_ref_norm', 'queue_utilization', 'cycles_wo_finds_norm',
    'coverage_density', 'time_since_find_norm', 'reserved_0', 'reserved_1',
]


@dataclass
class TrainingRecord:
    entry_id: int
    energy_assigned: int
    new_edges: int
    new_hit_counts: int
    found_crash: bool
    found_hang: bool
    features: np.ndarray   # shape (16,)
    timestamp_us: int

    @property
    def reward(self) -> float:
        """Reward функция — идентична C-коду в ml-scheduler.c"""
        return (10.0 * self.new_edges
                + 0.1 * self.new_hit_counts
                + 50.0 * self.found_crash
                + 20.0 * self.found_hang)


def parse_log(path: str, offset: int = 0) -> List[TrainingRecord]:
    """
    Читает все записи из лога начиная с byte-offset.
    Возвращает список записей и новый offset.
    Неполные записи в конце файла игнорируются.
    """
    records = []
    try:
        with open(path, 'rb') as f:
            f.seek(offset)
            while True:
                raw = f.read(RECORD_SIZE)
                if len(raw) < RECORD_SIZE:
                    break
                fields = struct.unpack(RECORD_FORMAT, raw)
                entry_id, energy, new_edges, new_hit, crash, hang = fields[:6]
                feats = np.array(fields[6:22], dtype=np.float32)
                ts = fields[22]
                records.append(TrainingRecord(
                    entry_id=entry_id,
                    energy_assigned=energy,
                    new_edges=new_edges,
                    new_hit_counts=new_hit,
                    found_crash=bool(crash),
                    found_hang=bool(hang),
                    features=feats,
                    timestamp_us=ts,
                ))
    except FileNotFoundError:
        pass
    return records


def records_to_arrays(records: List[TrainingRecord]):
    """Конвертирует список записей в X (features) и y (reward)."""
    if not records:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.float32)
    X = np.stack([r.features for r in records])
    y = np.array([r.reward for r in records], dtype=np.float32)
    return X, y
