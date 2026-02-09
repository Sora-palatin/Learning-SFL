"""
Simple logging helper for SFL Contract project.
"""
import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
