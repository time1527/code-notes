import logging
import time
import os


def setup_logger(level=logging.INFO):
    # Do not run handler.setLevel(level) so that users can change the level via logger.setLevel later
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    os.makedirs("logs", exist_ok=True)

    file_handler = logging.FileHandler(f"logs/{time.time()}.log")
    file_handler.setFormatter(formatter)

    _logger = logging.getLogger("vit_logger")
    _logger.setLevel(level)
    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)
    return _logger
