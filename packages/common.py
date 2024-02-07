import logging

GLOBAL_LOGGING_LEVEL = logging.DEBUG

def create_custom_logger(name: str, level=logging.DEBUG):
    """Creates a logger with the given name and level.
    If the logger already exists, it will be returned, otherwise it will be created.

    Args:
        name (str): The name of the logger
        level (int, optional): The logger's filter level. Defaults to logging.DEBUG.

    Returns:
        Logger: The logger
    """
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(max(level, GLOBAL_LOGGING_LEVEL))
    return logger