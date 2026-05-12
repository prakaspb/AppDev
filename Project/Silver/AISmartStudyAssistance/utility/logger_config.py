import logging
import os
from datetime import datetime


def setup_logger():
    # Get current execution path
    base_path = os.getcwd()

    # Create logs folder under current path
    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Timestamped log file
    log_file = os.path.join(
        log_dir,
        f"study_buddy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ],
    )

    logging.info(f"Logger initialized. Log file: {log_file}")