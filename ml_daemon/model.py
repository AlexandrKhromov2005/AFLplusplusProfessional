"""
ML-модель для предсказания energy (power schedule).

Архитектура: двухуровневая.
  Уровень 1: базовая модель — GradientBoostingRegressor на всех данных.
  Уровень 2: онлайн-адаптация — Ridge regression на последних N записях
             поверх признаков от уровня 1.

Почему GBT а не нейросеть:
  - Таблица 16 признаков, сотни/тысячи записей — классика для деревьев
  - Быстрый inference (<0.1ms), не нужен GPU
  - Нет зависимости от PyTorch/ONNX — только numpy+sklearn
"""

import numpy as np
import pickle
import os
import threading
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Optional

# Диапазон допустимых energy значений
ENERGY_MIN = 1
ENERGY_MAX = 1600
ENERGY_DEFAULT = 100  # возвращаем если нет данных

# Минимум записей для первого обучения базовой модели
MIN_RECORDS_BASE = 50
# Минимум записей для онлайн-адаптации
MIN_RECORDS_ONLINE = 10
# Размер скользящего окна для онлайн-адаптации
ONLINE_WINDOW = 500


class EnergyModel:
    """
    Предсказывает energy для queue_entry по его feature-вектору.
    """

    def __init__(self):
        self.base_model: Optional[GradientBoostingRegressor] = None
        self.online_model: Optional[Ridge] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.total_trained = 0
        self.update_count = 0
        # Lock защищает scaler/base_model/online_model от concurrent access
        # между fit_base (trainer thread) и predict (server thread)
        self._lock = threading.Lock()

        # Скользящее окно последних ONLINE_WINDOW записей (numpy arrays)
        self._window_X = np.empty((0, 16), dtype=np.float32)
        self._window_y = np.empty(0, dtype=np.float32)

    def _clip_energy(self, val: float) -> int:
        """Ограничить предсказание допустимым диапазоном."""
        return int(np.clip(round(val), ENERGY_MIN, ENERGY_MAX))

    def fit_base(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучить базовую модель на всём накопленном датасете.
        Вызывается при достижении MIN_RECORDS_BASE записей
        и затем каждые REFIT_BASE_EVERY новых записей.
        """
        if len(X) < MIN_RECORDS_BASE:
            return

        # Обучить всё в локальных переменных ВНЕ lock —
        # чтобы predict() не блокировался на время GBT fit (~50ms)
        new_scaler = StandardScaler()
        new_scaler.fit(X)
        X_scaled = new_scaler.transform(X)

        new_base = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        new_base.fit(X_scaled, y)

        # Атомарная замена под lock — predict() увидит консистентное состояние
        with self._lock:
            self.scaler = new_scaler
            self.base_model = new_base
            self.online_model = None  # stale Ridge после нового scaler
            self.total_trained = len(X)
            self.is_fitted = True

        print(f"[MLF daemon] Base model fitted on {len(X)} samples")

    def update_online(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Онлайн-адаптация: Ridge regression на скользящем окне.
        Вызывается после каждого батча новых записей.
        """
        # Обновить скользящее окно (numpy arrays — без list→array конверсии)
        self._window_X = np.vstack([self._window_X, X_new.astype(np.float32)])
        self._window_y = np.concatenate([self._window_y, y_new.astype(np.float32)])
        if len(self._window_X) > ONLINE_WINDOW:
            self._window_X = self._window_X[-ONLINE_WINDOW:]
            self._window_y = self._window_y[-ONLINE_WINDOW:]

        if len(self._window_X) < MIN_RECORDS_ONLINE:
            return
        if not self.is_fitted:
            return

        # Снять snapshot scaler/base_model под lock
        with self._lock:
            scaler = self.scaler
            base = self.base_model

        X_scaled = scaler.transform(self._window_X)
        base_pred = base.predict(X_scaled).reshape(-1, 1)
        X_aug = np.hstack([X_scaled, base_pred])

        new_model = Ridge(alpha=1.0)
        new_model.fit(X_aug, self._window_y)

        with self._lock:
            self.online_model = new_model
        self.update_count += 1

    def predict(self, features: np.ndarray) -> int:
        """
        Предсказать energy для одного feature-вектора (shape: (16,)).
        Возвращает int в диапазоне [ENERGY_MIN, ENERGY_MAX].
        Если модель не обучена — возвращает ENERGY_DEFAULT.
        """
        # Снять консистентный snapshot под lock
        with self._lock:
            if not self.is_fitted or self.base_model is None:
                return ENERGY_DEFAULT
            scaler = self.scaler
            base = self.base_model
            online = self.online_model

        X = features.reshape(1, -1).astype(np.float32)
        X_scaled = scaler.transform(X)

        if online is not None and hasattr(online, 'coef_'):
            base_pred = base.predict(X_scaled).reshape(-1, 1)
            X_aug = np.hstack([X_scaled, base_pred])
            energy_raw = online.predict(X_aug)[0]
        else:
            energy_raw = base.predict(X_scaled)[0]

        return self._clip_energy(energy_raw)

    def save(self, path: str) -> None:
        """Сохранить модель на диск."""
        with open(path, 'wb') as f:
            pickle.dump({
                'base_model': self.base_model,
                'online_model': self.online_model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'total_trained': self.total_trained,
                'update_count': self.update_count,
                '_window_X': self._window_X[-ONLINE_WINDOW:],
                '_window_y': self._window_y[-ONLINE_WINDOW:],
            }, f)

    def load(self, path: str) -> bool:
        """Загрузить модель с диска. Возвращает True если успешно."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.base_model = state['base_model']
            self.online_model = state['online_model']
            self.scaler = state['scaler']
            self.is_fitted = state['is_fitted']
            self.total_trained = state['total_trained']
            self.update_count = state['update_count']
            # Поддержка старых чекпоинтов где _window_X был list
            raw_X = state.get('_window_X', np.empty((0, 16), dtype=np.float32))
            raw_y = state.get('_window_y', np.empty(0, dtype=np.float32))
            if isinstance(raw_X, list):
                self._window_X = np.array(raw_X, dtype=np.float32).reshape(-1, 16) if raw_X else np.empty((0, 16), dtype=np.float32)
                self._window_y = np.array(raw_y, dtype=np.float32)
            else:
                self._window_X = raw_X
                self._window_y = raw_y
            print(f"[MLF daemon] Model loaded from {path} "
                  f"(trained={self.total_trained}, updates={self.update_count})")
            return True
        except Exception as e:
            print(f"[MLF daemon] Failed to load model: {e}")
            return False
