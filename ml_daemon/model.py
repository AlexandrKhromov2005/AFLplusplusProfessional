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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

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

        # Скользящее окно последних ONLINE_WINDOW записей
        self._window_X = []
        self._window_y = []

    def _clip_energy(self, val: float) -> int:
        """Ограничить предсказание допустимым диапазоном."""
        return int(np.clip(round(val), ENERGY_MIN, ENERGY_MAX))

    def fit_base(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучить базовую модель на всём накопленном датасете.
        Вызывается при достижении MIN_RECORDS_BASE записей
        и затем каждые 1000 новых записей.
        """
        if len(X) < MIN_RECORDS_BASE:
            return

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.base_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self.base_model.fit(X_scaled, y)
        self.total_trained = len(X)
        self.is_fitted = True
        print(f"[MLF daemon] Base model fitted on {len(X)} samples")

    def update_online(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Онлайн-адаптация: Ridge regression на скользящем окне.
        Вызывается после каждого батча новых записей.
        """
        # Обновить скользящее окно
        self._window_X.extend(X_new.tolist())
        self._window_y.extend(y_new.tolist())
        if len(self._window_X) > ONLINE_WINDOW:
            self._window_X = self._window_X[-ONLINE_WINDOW:]
            self._window_y = self._window_y[-ONLINE_WINDOW:]

        if len(self._window_X) < MIN_RECORDS_ONLINE:
            return
        if not self.is_fitted:
            return

        X_win = np.array(self._window_X, dtype=np.float32)
        y_win = np.array(self._window_y, dtype=np.float32)
        X_scaled = self.scaler.transform(X_win)

        # Используем base_model predictions как признак для online модели
        base_pred = self.base_model.predict(X_scaled).reshape(-1, 1)
        X_aug = np.hstack([X_scaled, base_pred])

        self.online_model = Ridge(alpha=1.0)
        self.online_model.fit(X_aug, y_win)
        self.update_count += 1

    def predict(self, features: np.ndarray) -> int:
        """
        Предсказать energy для одного feature-вектора (shape: (16,)).
        Возвращает int в диапазоне [ENERGY_MIN, ENERGY_MAX].
        Если модель не обучена — возвращает ENERGY_DEFAULT.
        """
        if not self.is_fitted or self.base_model is None:
            return ENERGY_DEFAULT

        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        if self.online_model is not None and hasattr(self.online_model, 'coef_'):
            base_pred = self.base_model.predict(X_scaled).reshape(-1, 1)
            X_aug = np.hstack([X_scaled, base_pred])
            energy_raw = self.online_model.predict(X_aug)[0]
        else:
            energy_raw = self.base_model.predict(X_scaled)[0]

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
            self._window_X = state.get('_window_X', [])
            self._window_y = state.get('_window_y', [])
            print(f"[MLF daemon] Model loaded from {path} "
                  f"(trained={self.total_trained}, updates={self.update_count})")
            return True
        except Exception as e:
            print(f"[MLF daemon] Failed to load model: {e}")
            return False
