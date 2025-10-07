import logging
import os


def setup_logger(log_dir, log_file, name="default"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita handlers duplicados
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
    