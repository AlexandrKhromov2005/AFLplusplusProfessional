# AFLplusplusProfessional

AFL++ 4.32c со встроенным **ML-планировщиком мутаций** (`mlf-fuzz`).

ML-планировщик заменяет стандартный `calculate_score()` в AFL++ на обученную модель, которая предсказывает, сколько мутаций назначить каждому элементу очереди. Модель обучается онлайн в фоне во время фаззинга — предварительное обучение не требуется.

---

## Архитектура

```
mlf-fuzz (C)                      ml_daemon (Python)
┌─────────────────────────┐       ┌──────────────────────────────┐
│  calculate_score()      │──────▶│  SchedulerServer             │
│  отправляет 16 float32  │  UNIX │  принимает признаки          │
│  признаков на элемент   │  sock │  возвращает int32 energy     │
│  очереди (~64 байта)    │◀──────│                              │
│                         │       │  OnlineTrainer               │
│  пишет тренировочный    │──────▶│  опрашивает лог каждые 5с    │
│  лог (~96 байт/запись)  │  file │  обучает базовую GBT (≥50)   │
└─────────────────────────┘       │  обновляет Ridge онлайн-слой │
                                  └──────────────────────────────┘
```

- **Сокет:** `/tmp/mlf_scheduler.sock` (env `MLF_SCHEDULER_SOCKET`)
- **Тренировочный лог:** `/tmp/mlf_training.bin` (env `MLF_TRAINING_LOG`)
- **Чекпоинт модели:** `/tmp/mlf_model.pkl`
- **Fallback:** если демон недоступен, автоматически используется стандартный планировщик AFL++

---

## Требования

- Linux x86-64
- AFL++ 4.32c (включён в `AFLplusplus/`)
- Clang/LLVM с `afl-clang-fast`
- Python 3.10+, `numpy`, `scikit-learn`

```bash
pip3 install numpy scikit-learn
```

---

## Сборка

### 1. Собрать `mlf-fuzz`

```bash
cd AFLplusplus
make mlf-fuzz -j$(nproc)
```

Результат: `AFLplusplus/mlf-fuzz`

### 2. Собрать таргет для фаззинга

Исходники клонируются в `targets/src/`, бинарники попадают в `targets/bin/`.

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

## Использование

### Шаг 1 — Запустить ML-демон

```bash
cd /path/to/AFLplusplusProfessional
python3 -m ml_daemon.daemon \
    --socket /tmp/mlf_scheduler.sock \
    --log    /tmp/mlf_training.bin \
    --model  /tmp/mlf_model.pkl
```

Демон стартует сразу и возвращает `energy=100` (дефолт), пока не накоплено ≥ 50 записей. После этого обучается базовая GBT-модель и предсказания становятся реальными.

### Шаг 2 — Запустить `mlf-fuzz`

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

| Флаг | Значение |
| --- | --- |
| `-i seeds/` | Входной корпус (8 JSON-сидов включены) |
| `-o /tmp/out_yyjson` | Выходная директория (создаётся автоматически) |
| `-m none` | Отключить лимит памяти (нужно при ASAN) |
| `-t 10000` | Таймаут на один запуск, мс |
| `-x json.dict` | Словарь токенов (26 JSON-токенов включены) |

### Запуск без ML (fallback / baseline)

```bash
AFL_SKIP_CPUFREQ=1 MLF_SCHEDULER_DISABLE=1 \
./AFLplusplus/mlf-fuzz -i seeds/ -o /tmp/out_base -m none -t 10000 \
    -- ./targets/bin/fuzz_yyjson
```

`MLF_SCHEDULER_DISABLE=1` отключает ML-хук — mlf-fuzz ведёт себя идентично стандартному `afl-fuzz`.

---

## Жизненный цикл ML-модели

| Фаза | Условие | Поведение |
| --- | --- | --- |
| Холодный старт | < 50 записей | Возвращает `ENERGY_DEFAULT = 100` |
| Базовая модель | ≥ 50 записей | GBT обучается на всех данных |
| Онлайн-адаптация | Каждая новая партия | Ridge-регрессия на скользящем окне 500 записей |
| Переобучение базы | Каждые 500 новых записей | Полный refit GBT |

Модель сохраняется по пути `--model` при каждом refit и при завершении демона (SIGINT/SIGTERM).

---

## Структура проекта

```
AFLplusplusProfessional/
├── AFLplusplus/              # Исходники AFL++ 4.32c (модифицированы)
│   ├── src/
│   │   ├── afl-fuzz-queue.c  # calculate_score() с ML-хуком
│   │   └── ml-scheduler.c   # C-клиент сокета + запись тренировочного лога
│   ├── include/
│   │   └── ml-scheduler.h   # Константы протокола
│   └── mlf-fuzz              # Собранный бинарь (после make mlf-fuzz)
│
├── ml_daemon/                # Python ML-демон
│   ├── daemon.py             # Точка входа
│   ├── server.py             # UNIX-сокет сервер (threading)
│   ├── trainer.py            # Цикл онлайн-обучения
│   ├── model.py              # EnergyModel: GBT + Ridge
│   ├── log_reader.py         # Парсер бинарного лога (96 байт/запись)
│   └── test_daemon.py        # Юнит-тесты (7/7)
│
├── targets/
│   ├── fuzz_yyjson.c         # Харнес для yyjson
│   ├── fuzz_cjson.c          # Харнес для cJSON
│   ├── fuzz_tunnuz.cpp       # Харнес для tunnuz/json
│   ├── bin/                  # Собранные бинарники харнесов (в .gitignore)
│   └── src/                  # Исходники парсеров (в .gitignore)
│
├── seeds/                    # 8 JSON-сидов
└── json.dict                 # Словарь AFL++ (26 JSON-токенов)
```

---

## Запуск тестов

```bash
cd /path/to/AFLplusplusProfessional
python3 -m ml_daemon.test_daemon
```

Ожидаемый вывод:
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

## Переменные окружения

| Переменная | По умолчанию | Описание |
| --- | --- | --- |
| `MLF_SCHEDULER_SOCKET` | `/tmp/mlf_scheduler.sock` | Путь к UNIX-сокету |
| `MLF_TRAINING_LOG` | `/tmp/mlf_training.bin` | Путь к бинарному тренировочному логу |
| `MLF_SCHEDULER_DISABLE` | не задана | Установить `1` для отключения ML, используется стандартный планировщик |
