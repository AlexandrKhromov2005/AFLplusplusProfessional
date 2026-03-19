"""
UNIX-сокет сервер.
Принимает feature-векторы от mlf-fuzz, возвращает energy.
"""

import socket
import struct
import os
import threading
import time
import numpy as np
from typing import Optional
from .model import EnergyModel

FEATURE_SIZE = 16 * 4   # 16 float32 = 64 байта
RESPONSE_SIZE = 4        # int32


class SchedulerServer:
    """
    Слушает на UNIX-сокете, обрабатывает запросы от mlf-fuzz.
    Поток-безопасен: model обновляется через lock.
    """

    def __init__(self, socket_path: str, model: EnergyModel):
        self.socket_path = socket_path
        self.model = model
        self.lock = threading.Lock()
        self._stop = threading.Event()

        # Статистика
        self.requests_total = 0
        self.requests_ml = 0
        self.requests_default = 0
        self.last_request_time = 0.0

    def update_model(self, new_model: EnergyModel) -> None:
        """Атомарно заменить модель (вызывается из потока обучения)."""
        with self.lock:
            self.model = new_model

    def _handle_client(self, conn: socket.socket, addr) -> None:
        """Обработчик одного подключения (mlf-fuzz держит соединение открытым)."""
        conn.settimeout(1.0)
        try:
            while not self._stop.is_set():
                # Читаем ровно FEATURE_SIZE байт
                data = b''
                while len(data) < FEATURE_SIZE:
                    try:
                        chunk = conn.recv(FEATURE_SIZE - len(data))
                    except socket.timeout:
                        # Таймаут — проверить stop флаг и продолжить
                        if self._stop.is_set():
                            return
                        continue
                    if not chunk:
                        return  # соединение закрыто
                    data += chunk

                # Десериализовать features
                floats = struct.unpack('16f', data)
                features = np.array(floats, dtype=np.float32)

                # Предсказать energy
                with self.lock:
                    energy = self.model.predict(features)

                # Отправить ответ
                response = struct.pack('i', energy)
                conn.sendall(response)

                self.requests_total += 1
                self.last_request_time = time.time()
                if energy > 0:
                    self.requests_ml += 1
                else:
                    self.requests_default += 1

        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            print(f"[MLF daemon] Client error: {e}")
        finally:
            conn.close()

    def serve_forever(self) -> None:
        """Главный цикл сервера. Блокирует поток."""
        # Удалить старый сокет если есть
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.socket_path)
        srv.listen(5)
        srv.settimeout(1.0)

        print(f"[MLF daemon] Listening on {self.socket_path}")

        try:
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                    t = threading.Thread(
                        target=self._handle_client,
                        args=(conn, addr),
                        daemon=True,
                    )
                    t.start()
                except socket.timeout:
                    continue
        finally:
            srv.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def stop(self) -> None:
        self._stop.set()
