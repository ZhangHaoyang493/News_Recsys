import logging
import sys

class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class Logger:
    @staticmethod
    def get_logger(name: str, level=logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Prevent adding multiple handlers if get_logger is called multiple times
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter())
            logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
        
        return logger
