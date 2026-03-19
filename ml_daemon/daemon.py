"""
Точка входа ML-демона.
Запускает три компонента:
  1. OnlineTrainer (поток чтения лога и обучения)
  2. SchedulerServer (UNIX-сокет сервер, главный поток)

Использование:
  python3 -m ml_daemon.daemon [опции]
"""

import argparse
import signal
import sys
import threading
import time

from .model import EnergyModel
from .server import SchedulerServer
from .trainer import OnlineTrainer


def main():
    parser = argparse.ArgumentParser(description='MLF Power Scheduler Daemon')
    parser.add_argument('--socket', default='/tmp/mlf_scheduler.sock',
                        help='UNIX socket path (default: /tmp/mlf_scheduler.sock)')
    parser.add_argument('--log', default='/tmp/mlf_training.bin',
                        help='Training log path (default: /tmp/mlf_training.bin)')
    parser.add_argument('--model', default='/tmp/mlf_model.pkl',
                        help='Model checkpoint path (default: /tmp/mlf_model.pkl)')
    parser.add_argument('--poll', type=float, default=5.0,
                        help='Log poll interval in seconds (default: 5.0)')
    args = parser.parse_args()

    print(f"[MLF daemon] Starting")
    print(f"[MLF daemon]   socket: {args.socket}")
    print(f"[MLF daemon]   log:    {args.log}")
    print(f"[MLF daemon]   model:  {args.model}")

    # Инициализировать модель
    model = EnergyModel()
    model.load(args.model)  # загрузить если есть checkpoint

    # Создать сервер
    server = SchedulerServer(
        socket_path=args.socket,
        model=model,
    )

    # Создать тренер
    trainer = OnlineTrainer(
        log_path=args.log,
        model=model,
        on_model_updated=server.update_model,
    )
    trainer.poll_interval = args.poll

    # Запустить тренер в фоновом потоке
    trainer_thread = threading.Thread(target=trainer.run, daemon=True)
    trainer_thread.start()

    # Graceful shutdown по Ctrl-C / SIGTERM
    def shutdown(sig, frame):
        print(f"\n[MLF daemon] Shutting down...")
        trainer.stop()
        server.stop()
        model.save(args.model)
        print(f"[MLF daemon] Model saved to {args.model}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Запустить сервер (блокирует главный поток)
    server.serve_forever()


if __name__ == '__main__':
    main()
