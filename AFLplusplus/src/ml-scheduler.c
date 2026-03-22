/*
 * AFLplusplusProfessional — ML Power Scheduler implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

#include "afl-fuzz.h"
#include "ml-scheduler.h"

/* --- Внутренние хелперы --- */

static uint64_t get_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

static float safe_log2(float x) {
  if (x <= 0) return 0.0f;
  return log2f(x);
}

/* --- Публичный API --- */

ml_sched_state_t *ml_sched_init(const char *socket_path, const char *log_path) {

  ml_sched_state_t *ms = calloc(1, sizeof(ml_sched_state_t));
  if (!ms) return NULL;

  ms->socket_fd = -1;
  ms->daemon_available = 0;

  /* Путь к сокету */
  if (socket_path) {
    strncpy(ms->socket_path, socket_path, sizeof(ms->socket_path) - 1);
  } else {
    strncpy(ms->socket_path, ML_SOCKET_PATH_DEFAULT, sizeof(ms->socket_path) - 1);
  }

  /* Открыть лог-файл */
  const char *lp = log_path ? log_path : "/tmp/mlf_training.bin";
  ms->training_log = fopen(lp, "ab");
  if (!ms->training_log) {
    fprintf(stderr, "[MLF] Warning: cannot open training log %s: %s\n",
            lp, strerror(errno));
  }

  /* Попытка подключиться к демону */
  ms->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (ms->socket_fd >= 0) {

    /* Неблокирующий режим */
    int flags = fcntl(ms->socket_fd, F_GETFL, 0);
    fcntl(ms->socket_fd, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, ms->socket_path, sizeof(addr.sun_path) - 1);

    if (connect(ms->socket_fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      ms->daemon_available = 1;
      fprintf(stderr, "[MLF] Connected to ML daemon at %s\n", ms->socket_path);
    } else {
      close(ms->socket_fd);
      ms->socket_fd = -1;
      fprintf(stderr, "[MLF] ML daemon not available at %s (will use AFL fallback)\n",
              ms->socket_path);
    }

  }

  /* Stagnation re-evaluation defaults */
  ms->stagnation_mode = 0;
  ms->stagnation_threshold_ms = 300000ULL;  /* 5 минут */
  ms->stagnation_entered_at = 0;
  ms->stagnation_revisit_count = 0;
  ms->ml_total_energy_sum = 0;

  /* Env override: MLF_STAGNATION_SEC */
  const char *st = getenv("MLF_STAGNATION_SEC");
  if (st) ms->stagnation_threshold_ms = (uint64_t)atoi(st) * 1000ULL;

  /* Per-seed decay */
  ms->decay_rate = 0.5f;
  ms->decay_half_life = 500;
  ms->decay_min_energy = 10;
  ms->total_decayed = 0;
  ms->total_reward_resets = 0;

  const char *dr = getenv("MLF_DECAY_RATE");
  if (dr) ms->decay_rate = atof(dr);
  const char *dh = getenv("MLF_DECAY_HALFLIFE");
  if (dh) ms->decay_half_life = (uint32_t)atoi(dh);
  const char *dm = getenv("MLF_DECAY_MIN");
  if (dm) ms->decay_min_energy = (uint32_t)atoi(dm);

  return ms;

}

void ml_sched_destroy(ml_sched_state_t *ms) {
  if (!ms) return;
  if (ms->socket_fd >= 0) close(ms->socket_fd);
  if (ms->training_log) fclose(ms->training_log);
  free(ms);
}

void ml_build_features(afl_state_t *afl, struct queue_entry *q, ml_features_t *feat) {

  memset(feat, 0, sizeof(ml_features_t));

  /* Средние значения из afl_state */
  float avg_exec_us = afl->total_cal_cycles > 0
      ? (float)afl->total_cal_us / afl->total_cal_cycles
      : 1000.0f;

  float avg_bitmap_size = afl->total_bitmap_entries > 0
      ? (float)afl->total_bitmap_size / afl->total_bitmap_entries
      : 1.0f;

  /* --- Признаки из queue_entry --- */
  feat->exec_us_norm     = (float)q->exec_us / avg_exec_us;
  /* Клипирование: 0..5 */
  if (feat->exec_us_norm > 5.0f) feat->exec_us_norm = 5.0f;
  feat->exec_us_norm /= 5.0f;

  feat->bitmap_size_norm = (float)q->bitmap_size / avg_bitmap_size;
  if (feat->bitmap_size_norm > 5.0f) feat->bitmap_size_norm = 5.0f;
  feat->bitmap_size_norm /= 5.0f;

  feat->fuzz_level_norm  = safe_log2((float)q->fuzz_level + 1) / 20.0f;
  feat->depth_norm       = safe_log2((float)q->depth + 1) / 10.0f;
  feat->handicap_norm    = (float)(q->handicap > 10 ? 10 : q->handicap) / 10.0f;

  u32 n_fuzz_val = afl->n_fuzz ? afl->n_fuzz[q->n_fuzz_entry] : 0;
  feat->n_fuzz_norm      = safe_log2((float)n_fuzz_val + 1) / 30.0f;

  feat->favored      = q->favored ? 1.0f : 0.0f;
  feat->passed_det   = q->passed_det ? 1.0f : 0.0f;
  feat->is_ascii     = q->is_ascii ? 1.0f : 0.0f;
  feat->tc_ref_norm  = (float)(q->tc_ref > 100 ? 100 : q->tc_ref) / 100.0f;

  /* --- Глобальный контекст --- */
  feat->queue_utilization = afl->active_items > 0
      ? (float)afl->queued_items / afl->active_items : 1.0f;
  if (feat->queue_utilization > 2.0f) feat->queue_utilization = 2.0f;
  feat->queue_utilization /= 2.0f;

  feat->cycles_wo_finds_norm = safe_log2((float)afl->cycles_wo_finds + 1) / 20.0f;

  /* coverage_density: приближение через total_bitmap_size */
  if (afl->total_bitmap_entries > 0 && afl->fsrv.map_size > 0) {
    float max_possible = (float)afl->fsrv.map_size * afl->total_bitmap_entries;
    feat->coverage_density = (float)afl->total_bitmap_size / max_possible;
    if (feat->coverage_density > 1.0f) feat->coverage_density = 1.0f;
  }

  u64 now = get_us();
  u64 elapsed_find = afl->last_find_time > 0 ? (now / 1000 - afl->last_find_time) : 0;
  feat->time_since_find_norm = (float)elapsed_find / 3600000.0f;
  if (feat->time_since_find_norm > 1.0f) feat->time_since_find_norm = 1.0f;

}

int32_t ml_request_energy(ml_sched_state_t *ms, ml_features_t *feat) {

  if (!ms || !ms->daemon_available || ms->socket_fd < 0) return -1;

  /* Протокол: отправляем ML_FEATURE_DIM float'ов, получаем один int32 */
  ssize_t sent = send(ms->socket_fd, feat, sizeof(ml_features_t), MSG_DONTWAIT);
  if (sent != (ssize_t)sizeof(ml_features_t)) {
    ms->daemon_timeouts++;
    if (ms->daemon_timeouts > 100) ms->daemon_available = 0;
    return -1;
  }

  /* Ждём ответ с таймаутом ML_SOCKET_TIMEOUT_US */
  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(ms->socket_fd, &rfds);
  struct timeval tv = {0, ML_SOCKET_TIMEOUT_US};

  int ready = select(ms->socket_fd + 1, &rfds, NULL, NULL, &tv);
  if (ready <= 0) {
    ms->daemon_timeouts++;
    if (ms->daemon_timeouts > 100) ms->daemon_available = 0;
    return -1;
  }

  int32_t energy = -1;
  ssize_t rcvd = recv(ms->socket_fd, &energy, sizeof(int32_t), 0);
  if (rcvd != (ssize_t)sizeof(int32_t) || energy <= 0) {
    ms->daemon_errors++;
    return -1;
  }

  ms->daemon_timeouts = 0; /* сбросить счётчик при успехе */
  return energy;

}

void ml_log_result(ml_sched_state_t *ms, uint32_t entry_id,
                   uint32_t energy, uint32_t new_edges,
                   uint32_t new_hit_counts, uint8_t crash, uint8_t hang,
                   ml_features_t *feat) {

  if (!ms || !ms->training_log) return;

  ml_training_record_t rec;
  memset(&rec, 0, sizeof(rec));
  rec.entry_id        = entry_id;
  rec.energy_assigned = energy;
  rec.new_edges       = new_edges;
  rec.new_hit_counts  = new_hit_counts;
  rec.found_crash     = crash;
  rec.found_hang      = hang;
  rec.features        = *feat;
  rec.timestamp_us    = get_us();

  fwrite(&rec, sizeof(rec), 1, ms->training_log);
  ms->log_records++;

  fflush(ms->training_log);

  /* Обновить EMA reward */
  float reward = 10.0f * new_edges + 0.1f * new_hit_counts
               + 50.0f * crash + 20.0f * hang;
  ms->last_reward = reward;
  ms->avg_reward_ema = 0.99f * ms->avg_reward_ema + 0.01f * reward;

}
