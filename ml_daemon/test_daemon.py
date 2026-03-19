"""
Тесты компонентов демона.
Запуск: python3 -m ml_daemon.test_daemon
"""

import struct
import tempfile
import os
import numpy as np
import threading
import socket
import time
import sys

from ml_daemon.log_reader import (
    RECORD_FORMAT, RECORD_SIZE, parse_log, records_to_arrays
)
from ml_daemon.model import EnergyModel, ENERGY_MIN, ENERGY_MAX, ENERGY_DEFAULT
from ml_daemon.server import SchedulerServer, FEATURE_SIZE


def make_fake_record(entry_id=0, energy=100, new_edges=3,
                     new_hit=1, crash=0, hang=0):
    """Создать синтетическую запись в бинарном формате."""
    features = [float(i) / 16.0 for i in range(16)]
    ts = 1234567890
    return struct.pack(RECORD_FORMAT,
                       entry_id, energy, new_edges, new_hit,
                       crash, hang,
                       *features, ts)


def test_record_size():
    """Проверить что размер записи ровно 96 байт."""
    assert RECORD_SIZE == 96, f"Wrong record size: {RECORD_SIZE}, expected 96"
    raw = make_fake_record()
    assert len(raw) == RECORD_SIZE, f"Packed size {len(raw)} != {RECORD_SIZE}"
    print("PASS: record size == 96")


def test_log_parser():
    """Проверить парсинг лога: запись → парсинг → проверка полей."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(make_fake_record(entry_id=42, energy=200, new_edges=5))
        f.write(make_fake_record(entry_id=43, energy=300, new_edges=0))
        path = f.name

    records = parse_log(path)
    assert len(records) == 2, f"Expected 2 records, got {len(records)}"
    assert records[0].entry_id == 42
    assert records[0].energy_assigned == 200
    assert records[0].new_edges == 5
    assert records[0].features.shape == (16,)
    assert records[1].entry_id == 43
    os.unlink(path)
    print("PASS: log parser")


def test_model_default():
    """Нетренированная модель должна возвращать ENERGY_DEFAULT."""
    m = EnergyModel()
    features = np.zeros(16, dtype=np.float32)
    energy = m.predict(features)
    assert energy == ENERGY_DEFAULT, f"Expected {ENERGY_DEFAULT}, got {energy}"
    print("PASS: model returns default when not fitted")


def test_model_fit():
    """Обученная модель должна возвращать значения в допустимом диапазоне."""
    m = EnergyModel()
    rng = np.random.default_rng(42)
    X = rng.random((100, 16)).astype(np.float32)
    y = rng.uniform(10, 500, 100).astype(np.float32)
    m.fit_base(X, y)

    assert m.is_fitted
    for _ in range(10):
        feat = rng.random(16).astype(np.float32)
        energy = m.predict(feat)
        assert ENERGY_MIN <= energy <= ENERGY_MAX, \
            f"Energy {energy} out of range [{ENERGY_MIN}, {ENERGY_MAX}]"
    print("PASS: model fit and predict in range")


def test_model_online_update():
    """Онлайн-обновление не должно ронять модель."""
    m = EnergyModel()
    rng = np.random.default_rng(0)
    X = rng.random((100, 16)).astype(np.float32)
    y = rng.uniform(50, 300, 100).astype(np.float32)
    m.fit_base(X, y)

    X_new = rng.random((20, 16)).astype(np.float32)
    y_new = rng.uniform(50, 300, 20).astype(np.float32)
    m.update_online(X_new, y_new)

    assert m.online_model is not None
    energy = m.predict(rng.random(16).astype(np.float32))
    assert ENERGY_MIN <= energy <= ENERGY_MAX
    print("PASS: online update")


def test_model_save_load():
    """Сохранение и загрузка модели."""
    m = EnergyModel()
    rng = np.random.default_rng(7)
    X = rng.random((60, 16)).astype(np.float32)
    y = rng.uniform(10, 400, 60).astype(np.float32)
    m.fit_base(X, y)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        path = f.name
    m.save(path)

    m2 = EnergyModel()
    ok = m2.load(path)
    assert ok and m2.is_fitted
    os.unlink(path)
    print("PASS: model save/load")


def test_server_socket():
    """Сервер должен принимать feature-вектор и возвращать int32."""
    m = EnergyModel()
    rng = np.random.default_rng(1)
    X = rng.random((60, 16)).astype(np.float32)
    y = rng.uniform(50, 400, 60).astype(np.float32)
    m.fit_base(X, y)

    sock_path = '/tmp/mlf_test_server.sock'
    srv = SchedulerServer(socket_path=sock_path, model=m)

    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)  # дать серверу запуститься

    try:
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cli.connect(sock_path)

        # Отправить feature-вектор
        features = rng.random(16).astype(np.float32)
        cli.sendall(features.tobytes())

        # Получить ответ
        resp = cli.recv(4)
        assert len(resp) == 4, f"Expected 4 bytes, got {len(resp)}"
        energy = struct.unpack('i', resp)[0]
        assert ENERGY_MIN <= energy <= ENERGY_MAX, \
            f"Energy {energy} out of range"
        cli.close()
        print(f"PASS: server socket (energy={energy})")
    finally:
        srv.stop()
        time.sleep(0.2)


if __name__ == '__main__':
    tests = [
        test_record_size,
        test_log_parser,
        test_model_default,
        test_model_fit,
        test_model_online_update,
        test_model_save_load,
        test_server_socket,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {len(tests)-failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
