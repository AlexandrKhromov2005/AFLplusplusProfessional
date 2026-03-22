/*
 * ML Power Scheduler — структуры и API
 * Часть AFLplusplusProfessional
 */

#ifndef _ML_SCHEDULER_H
#define _ML_SCHEDULER_H

#include <stdint.h>

/* Максимальный размер feature-вектора для одного queue_entry */
#define ML_FEATURE_DIM 16

/* Версия протокола обмена с Python-демоном */
#define ML_PROTO_VERSION 1

/* Путь к UNIX-сокету для связи с Python-демоном */
#define ML_SOCKET_PATH_DEFAULT "/tmp/mlf_scheduler.sock"

/* Таймаут ожидания ответа от демона (мкс) */
#define ML_SOCKET_TIMEOUT_US 5000

/*
 * Feature-вектор для одного queue_entry.
 * Поля должны быть нормализованы: все float в диапазоне [0, 1]
 * кроме помеченных явно.
 *
 * Соответствует полям struct queue_entry из include/afl-fuzz.h:
 */
typedef struct ml_features {

  /* --- Из queue_entry (динамические, AFL знает) --- */
  float exec_us_norm;       /* exec_us / avg_exec_us (из afl_state) */
  float bitmap_size_norm;   /* bitmap_size / avg_bitmap_size */
  float fuzz_level_norm;    /* log2(fuzz_level + 1) / 20.0 */
  float depth_norm;         /* log2(depth + 1) / 10.0 */
  float handicap_norm;      /* min(handicap, 10) / 10.0 */
  float n_fuzz_norm;        /* log2(n_fuzz[entry] + 1) / 30.0 */
  float favored;            /* 1.0 если favored, иначе 0.0 */
  float passed_det;         /* 1.0 если deterministic stages пройдены */
  float is_ascii;           /* 1.0 если текстовый input */
  float tc_ref_norm;        /* min(tc_ref, 100) / 100.0 */

  /* --- Глобальный контекст (из afl_state) --- */
  float queue_utilization;  /* queued_items / max(active_items, 1) — насколько очередь занята */
  float cycles_wo_finds_norm; /* log2(cycles_wo_finds + 1) / 20.0 */
  float coverage_density;   /* покрытых edges / общий размер bitmap */
  float time_since_find_norm; /* (now - last_find_time) / 3600000.0 (нормировано к 1ч) */

  /* --- Зарезервировано для будущих признаков --- */
  float reserved_0;
  float reserved_1;

} ml_features_t;

/*
 * Запись в лог для обучения.
 * Сохраняется после каждого завершённого fuzz_one() вызова.
 */
typedef struct ml_training_record {

  uint32_t entry_id;          /* queue_entry->id */
  uint32_t energy_assigned;   /* сколько мутаций назначил шедулер */
  uint32_t new_edges;         /* сколько новых edges нашли */
  uint32_t new_hit_counts;    /* сколько новых hit counts */
  uint8_t  found_crash;       /* 1 если нашли краш */
  uint8_t  found_hang;        /* 1 если нашли hang */
  ml_features_t features;     /* признаки на момент назначения */
  uint64_t timestamp_us;      /* время записи */

} ml_training_record_t;

/*
 * Состояние ML-шедулера, хранится в afl_state->ml_sched
 */
typedef struct ml_sched_state {

  /* Подключение к Python-демону */
  int      socket_fd;              /* -1 если не подключён */
  char     socket_path[256];       /* путь к UNIX-сокету */
  uint8_t  daemon_available;       /* 1 если демон отвечает */
  uint32_t daemon_timeouts;        /* счётчик таймаутов */
  uint32_t daemon_errors;          /* счётчик ошибок */

  /* Статистика для UI */
  uint64_t ml_total_decisions;     /* сколько раз ML принял решение */
  uint64_t afl_total_decisions;    /* сколько раз использовался AFL-fallback */
  float    last_reward;            /* reward последней записи */
  float    avg_reward_ema;         /* EMA наград (alpha=0.01) */
  float    last_entropy;           /* энтропия последних предсказаний */
  uint32_t model_updates;          /* сколько раз модель обновлялась */
  uint32_t collapse_warnings;      /* счётчик предупреждений о коллапсе */

  /* Лог-файл для обучения */
  FILE    *training_log;           /* FILE* к training_log.bin */
  uint64_t log_records;            /* кол-во записанных записей */

  /* Stagnation-triggered re-evaluation */
  uint8_t  stagnation_mode;           /* 1 = в режиме ревизии */
  uint64_t stagnation_threshold_ms;   /* Порог: мс без находок (default 300000) */
  uint64_t stagnation_entered_at;     /* Когда вошли в режим (ms) */
  uint32_t stagnation_revisit_count;  /* Сколько забытых сидов пересмотрели */
  uint64_t ml_total_energy_sum;       /* Суммарный energy всех ML-решений */

} ml_sched_state_t;

/* --- Публичный API --- */

/* Инициализация: открыть лог, подключиться к демону (если доступен) */
ml_sched_state_t *ml_sched_init(const char *socket_path, const char *log_path);

/* Освобождение ресурсов */
void ml_sched_destroy(ml_sched_state_t *ms);

/* Построить feature-вектор из queue_entry + afl_state */
void ml_build_features(afl_state_t *afl, struct queue_entry *q, ml_features_t *feat);

/*
 * Запросить energy у ML-демона.
 * Возвращает предсказанное значение или -1 при ошибке/таймауте.
 * При ошибке вызывающий код использует AFL-шедулер как fallback.
 */
int32_t ml_request_energy(ml_sched_state_t *ms, ml_features_t *feat);

/* Записать результат в лог для обучения */
void ml_log_result(ml_sched_state_t *ms, uint32_t entry_id,
                   uint32_t energy, uint32_t new_edges,
                   uint32_t new_hit_counts, uint8_t crash, uint8_t hang,
                   ml_features_t *feat);

#endif /* _ML_SCHEDULER_H */
