import logging

GLOBAL_LOGGING_LEVEL = logging.DEBUG

def create_custom_logger(name: str, level=logging.DEBUG):
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(max(level, GLOBAL_LOGGING_LEVEL))
    return logger