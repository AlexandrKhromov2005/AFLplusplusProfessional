"""
Тест воспроизведения race condition в EnergyModel.
Запуск: python3 -m ml_daemon.test_race
"""
import numpy as np
import threading
import time
import traceback
from ml_daemon.model import EnergyModel

def test_reproduce_race():
    """
    Симулируем то что происходит в реальности:
    - trainer поток вызывает fit_base() и update_online()
    - server поток параллельно вызывает predict()
    - ошибка должна воспроизвестись здесь
    """
    model = EnergyModel()
    errors = []
    stop = threading.Event()

    # Поток 1: постоянно predict() как сервер
    def server_thread():
        while not stop.is_set():
            try:
                feat = np.random.rand(16).astype(np.float32)
                model.predict(feat)
            except Exception as e:
                errors.append(f"predict() error: {e}\n{traceback.format_exc()}")

    # Поток 2: обучение как trainer
    def trainer_thread():
        rng = np.random.default_rng(42)

        # Накопить данные (200 записей достаточно для воспроизведения)
        all_X = rng.random((200, 16)).astype(np.float32)
        all_y = rng.uniform(50, 500, 200).astype(np.float32)

        # Симулируем батчи как в реальном trainer
        for i in range(0, 200, 2):
            X_batch = all_X[i:i+2]
            y_batch = all_y[i:i+2]

            model.update_online(X_batch, y_batch)

            # fit_base на 50 записях, затем каждые 50
            total = i + 2
            if total >= 50 and (total == 50 or total % 50 == 0):
                X_all = all_X[:total]
                y_all = all_y[:total]
                model.fit_base(X_all, y_all)
                print(f"fit_base called at total={total}")

            time.sleep(0.001)  # небольшая пауза как в реальности

        stop.set()

    t_server = threading.Thread(target=server_thread, daemon=True)
    t_trainer = threading.Thread(target=trainer_thread)

    t_server.start()
    t_trainer.start()
    t_trainer.join(timeout=120)
    if t_trainer.is_alive():
        stop.set()
        t_trainer.join(timeout=5)
    else:
        stop.set()
    t_server.join(timeout=1)

    if errors:
        print(f"\nREPRODUCED {len(errors)} error(s):")
        print(errors[0])  # показать первую ошибку с traceback
        return False
    else:
        print("\nNo errors — race condition NOT reproduced")
        return True


if __name__ == '__main__':
    print("Testing race condition in EnergyModel...")
    result = test_reproduce_race()
    if result:
        print("PASS: No race condition detected")
    else:
        print("FAIL: Race condition exists")
