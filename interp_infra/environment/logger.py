"""Simple logging utility for environment operations."""

import sys
from enum import Enum


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3


class Logger:
    """Simple logger for environment operations."""

    level = LogLevel.INFO

    @classmethod
    def set_level(cls, level: LogLevel):
        cls.level = level

    @classmethod
    def debug(cls, msg: str):
        if cls.level.value <= LogLevel.DEBUG.value:
            print(f"[DEBUG] {msg}", file=sys.stderr)

    @classmethod
    def info(cls, msg: str):
        if cls.level.value <= LogLevel.INFO.value:
            print(msg, file=sys.stderr)

    @classmethod
    def warn(cls, msg: str):
        if cls.level.value <= LogLevel.WARN.value:
            print(f"Warning: {msg}", file=sys.stderr)

    @classmethod
    def error(cls, msg: str):
        if cls.level.value <= LogLevel.ERROR.value:
            print(f"Error: {msg}", file=sys.stderr)


log = Logger()
