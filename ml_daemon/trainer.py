"""
Поток онлайн-обучения.
Периодически читает новые записи из training_log и дообучает модель.
"""

import time
import threading
import numpy as np
from .log_reader import parse_log, records_to_arrays, TrainingRecord
from .model import EnergyModel, MIN_RECORDS_BASE

# Интервал между проверками лога (секунды)
POLL_INTERVAL = 5.0
# Переобучать базовую модель каждые N новых записей
REFIT_BASE_EVERY = 500


class OnlineTrainer:
    """
    Читает training_log, обучает модель, уведомляет сервер об обновлениях.
    """

    def __init__(self, log_path: str, model: EnergyModel,
                 on_model_updated=None):
        self.log_path = log_path
        self.model = model
        self.on_model_updated = on_model_updated  # callback → server.update_model
        self._stop = threading.Event()
        self.poll_interval = POLL_INTERVAL

        self._log_offset = 0          # байтовый offset в файле
        self._all_records = []         # все записи для базовой модели
        self._records_since_refit = 0  # новые записи с последнего refit
        self._last_new_edge_time = time.time()  # stagnation tracking

    def _maybe_refit_base(self) -> None:
        """Переобучить базовую модель если накопилось достаточно данных."""
        n = len(self._all_records)
        if n < MIN_RECORDS_BASE:
            return
        if (not self.model.is_fitted) or (self._records_since_refit >= REFIT_BASE_EVERY):
            X, y = records_to_arrays(self._all_records)
            self.model.fit_base(X, y)
            self._records_since_refit = 0
            self._save_model()

    def _save_model(self) -> None:
        try:
            self.model.save('/tmp/mlf_model.pkl')
        except Exception as e:
            print(f"[MLF trainer] Save failed: {e}")

    def _log_stats(self, new_count: int) -> None:
        total = len(self._all_records)
        fitted = self.model.is_fitted
        updates = self.model.update_count
        print(f"[MLF trainer] +{new_count} records | total={total} | "
              f"fitted={fitted} | online_updates={updates}")

    def run(self) -> None:
        """Главный цикл обучения. Блокирует поток."""
        print(f"[MLF trainer] Watching {self.log_path}")

        while not self._stop.is_set():
            # Читаем новые записи
            new_records = parse_log(self.log_path, self._log_offset)

            if new_records:
                # Обновить offset
                from .log_reader import RECORD_SIZE
                self._log_offset += len(new_records) * RECORD_SIZE

                # Добавить в общий датасет
                self._all_records.extend(new_records)
                self._records_since_refit += len(new_records)

                # Обновить таймер стагнации
                for r in new_records:
                    if r.new_edges > 0:
                        self._last_new_edge_time = time.time()

                # Онлайн-адаптация на новых данных
                X_new, y_new = records_to_arrays(new_records)

                # Stagnation-aware reward shaping: бонус забытым сидам
                elapsed = time.time() - self._last_new_edge_time
                if elapsed > 300:  # 5 мин без новых edges
                    energies = np.array([r.energy_assigned for r in new_records],
                                        dtype=np.float32)
                    median_e = np.median(energies) if len(energies) > 0 else 100
                    # Забытые сиды (energy < median/2) получают бонус reward,
                    # чтобы модель училась давать им больше энергии
                    bonus = np.where(energies < median_e / 2, 5.0, 0.0)
                    y_new = y_new + bonus

                self.model.update_online(X_new, y_new)

                # Обновить сервер (атомарно)
                if self.on_model_updated:
                    self.on_model_updated(self.model)

                # Проверить нужно ли переобучить базовую модель
                self._maybe_refit_base()

                self._log_stats(len(new_records))

            # Ждём следующего цикла
            self._stop.wait(self.poll_interval)

    def stop(self) -> None:
        self._stop.set()
