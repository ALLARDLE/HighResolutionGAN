from loguru import logger
import sys


class MyFilter:
    def __init__(self, level):
        self.level = level

    def set_to_debug_level(self):
        self.level = "DEBUG"

    def set_to_info_level(self):
        self.level = "INFO"

    def __call__(self, record):
        levelno = logger.level(self.level).no
        return record["level"].no >= levelno


def init_logger(debug=False):
    logger.remove()
    global log_filter
    log_filter = MyFilter("INFO")
    logger.opt(depth=1)
    # Forward logging to standard output
    logger.add(
        sys.stdout,
        enqueue=True,
        filter=log_filter,
        format="<blue>{time:YYYY-MM-DD HH:mm:ss}</> || {message}",
        level=0,
        backtrace=True, diagnose=True,
    )
    if debug:
        log_filter.set_to_debug_level()
    else:
        log_filter.set_to_info_level()


# Logging
INFO_LOGGING_FILE_NAME = "info.log"
DEBUG_LOGGING_FILE_NAME = "debug.log"


def set_log_file(path):
    logger.remove()
    logger.add(sys.stdout, enqueue=True, filter=log_filter, level=0)
    info_path = path + '/' + INFO_LOGGING_FILE_NAME
    debug_path = path + '/' + DEBUG_LOGGING_FILE_NAME
    logger.add(info_path, level="INFO")
    logger.add(debug_path, level="DEBUG")
