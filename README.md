# AFLplusplusProfessional

AFL++ 4.32c with an integrated **ML power scheduler** (`mlf-fuzz`).

The ML scheduler replaces AFL++'s default `calculate_score()` with a learned model that predicts how many mutations to assign to each queue entry. It trains online in the background while fuzzing — no pre-training required.

---

## Architecture

```
mlf-fuzz (C)                      ml_daemon (Python)
┌─────────────────────────┐       ┌──────────────────────────────┐
│  calculate_score()      │──────▶│  SchedulerServer             │
│  sends 16 float32       │  UNIX │  receives features           │
│  features per queue     │  sock │  returns int32 energy        │
│  entry (~64 bytes)      │◀──────│                              │
│                         │       │  OnlineTrainer               │
│  writes training log    │──────▶│  polls log every 5s          │
│  (~96 bytes/record)     │  file │  fits GBT base model (≥50)   │
└─────────────────────────┘       │  updates Ridge online layer  │
                                  └──────────────────────────────┘
```

- **Socket:** `/tmp/mlf_scheduler.sock` (env `MLF_SCHEDULER_SOCKET`)
- **Training log:** `/tmp/mlf_training.bin` (env `MLF_TRAINING_LOG`)
- **Model checkpoint:** `/tmp/mlf_model.pkl`
- **Fallback:** if daemon is unreachable, standard AFL++ scheduler is used automatically

---

## Requirements

- Linux x86-64
- AFL++ 4.32c (included in `AFLplusplus/`)
- Clang/LLVM with `afl-clang-fast`
- Python 3.10+, `numpy`, `scikit-learn`

```bash
pip3 install numpy scikit-learn
```

---

## Build

### 1. Build `mlf-fuzz`

```bash
cd AFLplusplus
make mlf-fuzz -j$(nproc)
```

Output: `AFLplusplus/mlf-fuzz`

### 2. Build a fuzzing target

Sources are cloned into `targets/src/`, binaries go to `targets/bin/`.

**yyjson:**
```bash
git clone --depth=1 https://github.com/ibireme/yyjson.git targets/src/yyjson
cd AFLplusplus
AFL_SKIP_CPUFREQ=1 ./afl-clang-fast -std=c11 -fsanitize=address,undefined -g \
    -I ../targets/src/yyjson/src \
    ../targets/fuzz_yyjson.c ../targets/src/yyjson/src/yyjson.c \
    -o ../targets/bin/fuzz_yyjson
```

**cJSON:**
```bash
git clone --depth=1 https://github.com/DaveGamble/cJSON.git targets/src/cJSON
cd AFLplusplus
AFL_SKIP_CPUFREQ=1 ./afl-clang-fast -std=c11 -fsanitize=address,undefined -g \
    -I ../targets/src/cJSON \
    ../targets/fuzz_cjson.c ../targets/src/cJSON/cJSON.c \
    -o ../targets/bin/fuzz_cjson
```

**tunnuz/json:**
```bash
git clone --depth=1 https://github.com/tunnuz/json.git targets/src/tunnuz_json
cd targets/src/tunnuz_json
mkdir -p build && cd build
bison -d ../json.yy -o json.tab.cc
flex  -o lex.yy.cc ../json.l
cd /path/to/AFLplusplusProfessional/AFLplusplus
AFL_SKIP_CPUFREQ=1 ./afl-clang-fast++ -std=c++11 -fsanitize=address,undefined -g \
    -Wno-pessimizing-move -Wno-unneeded-internal-declaration \
    -I ../targets/src/tunnuz_json \
    -I ../targets/src/tunnuz_json/build \
    ../targets/fuzz_tunnuz.cpp \
    ../targets/src/tunnuz_json/build/json.tab.cc \
    ../targets/src/tunnuz_json/build/lex.yy.cc \
    ../targets/src/tunnuz_json/json_st.cc \
    -o ../targets/bin/fuzz_tunnuz
```

---

## Usage

### Step 1 — Start the ML daemon

```bash
cd /path/to/AFLplusplusProfessional
python3 -m ml_daemon.daemon \
    --socket /tmp/mlf_scheduler.sock \
    --log    /tmp/mlf_training.bin \
    --model  /tmp/mlf_model.pkl
```

The daemon starts immediately and returns `energy=100` (default) until it has collected ≥ 50 training records, after which the base GBT model is trained and predictions become real.

### Step 2 — Run `mlf-fuzz`

```bash
AFL_SKIP_CPUFREQ=1 \
MLF_TRAINING_LOG=/tmp/mlf_training.bin \
./AFLplusplus/mlf-fuzz \
    -i seeds/ \
    -o /tmp/out_yyjson \
    -m none \
    -t 10000 \
    -x json.dict \
    -- ./targets/bin/fuzz_yyjson
```

| Flag | Meaning |
|------|---------|
| `-i seeds/` | Input corpus (8 JSON seeds included) |
| `-o /tmp/out_yyjson` | Output directory (created automatically) |
| `-m none` | Disable memory limit (required with ASAN) |
| `-t 10000` | Timeout per execution, ms |
| `-x json.dict` | Token dictionary (26 JSON tokens included) |

### Run without ML (fallback / baseline)

```bash
AFL_SKIP_CPUFREQ=1 MLF_SCHEDULER_DISABLE=1 \
./AFLplusplus/mlf-fuzz -i seeds/ -o /tmp/out_base -m none -t 10000 \
    -- ./targets/bin/fuzz_yyjson
```

Setting `MLF_SCHEDULER_DISABLE=1` disables the ML hook — mlf-fuzz behaves identically to standard `afl-fuzz`.

---

## ML Model Lifecycle

| Phase | Condition | Behavior |
|-------|-----------|----------|
| Cold start | < 50 records | Returns `ENERGY_DEFAULT = 100` |
| Base model | ≥ 50 records | GBT trained on all data |
| Online adaptation | Every new batch | Ridge regression on 500-record sliding window |
| Base refit | Every 500 new records | Full GBT retrain |

The model is saved to `--model` path on every refit and on daemon shutdown (SIGINT/SIGTERM).

---

## Project Structure

```
AFLplusplusProfessional/
├── AFLplusplus/              # AFL++ 4.32c source (modified)
│   ├── src/
│   │   ├── afl-fuzz-queue.c  # calculate_score() with ML hook
│   │   └── ml-scheduler.c   # C-side socket client + training log writer
│   ├── include/
│   │   └── ml-scheduler.h   # Protocol constants
│   └── mlf-fuzz              # Built binary (after make mlf-fuzz)
│
├── ml_daemon/                # Python ML daemon
│   ├── daemon.py             # Entry point
│   ├── server.py             # UNIX socket server (threading)
│   ├── trainer.py            # Online training loop
│   ├── model.py              # EnergyModel: GBT + Ridge
│   ├── log_reader.py         # Binary log parser (96 bytes/record)
│   └── test_daemon.py        # Unit tests (7/7)
│
├── targets/
│   ├── fuzz_yyjson.c         # Harness for yyjson
│   ├── fuzz_cjson.c          # Harness for cJSON
│   ├── fuzz_tunnuz.cpp       # Harness for tunnuz/json
│   ├── bin/                  # Built harness binaries (gitignored)
│   └── src/                  # Cloned parser sources (gitignored)
│
├── seeds/                    # 8 JSON seed files
├── json.dict                 # AFL++ dictionary (26 JSON tokens)
└── PROJECT_STATE.md          # Current component statuses and test results
```

---

## Running Tests

```bash
cd /path/to/AFLplusplusProfessional
python3 -m ml_daemon.test_daemon
```

Expected output:
```
PASS: record size == 96
PASS: log parser
PASS: model returns default when not fitted
PASS: model fit and predict in range
PASS: online update
PASS: model save/load
PASS: server socket (energy=270)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLF_SCHEDULER_SOCKET` | `/tmp/mlf_scheduler.sock` | UNIX socket path |
| `MLF_TRAINING_LOG` | `/tmp/mlf_training.bin` | Binary training log path |
| `MLF_SCHEDULER_DISABLE` | unset | Set to `1` to disable ML, use standard scheduler |
