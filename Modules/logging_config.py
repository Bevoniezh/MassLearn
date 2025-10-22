"""Central logging configuration for MassLearn."""
from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional, Sequence, Any, Dict

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(user)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_USER = "anonymous"


class UserContextFilter(logging.Filter):
    """Ensure every log record includes the current user identifier."""

    def __init__(self, default_user: str = DEFAULT_USER) -> None:
        super().__init__(name="user-context")
        self._user = default_user

    @property
    def user(self) -> str:
        return self._user

    def set_user(self, user: Optional[str]) -> None:
        self._user = user.strip() if user else DEFAULT_USER

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if not getattr(record, "user", None):
            record.user = self._user
        return True


class UserFileHandler(logging.Handler):
    """Handler writing entries to a per-user log file."""

    def __init__(self, directory: Path, encoding: str = "utf-8") -> None:
        super().__init__()
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._encoding = encoding

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            user = getattr(record, "user", DEFAULT_USER) or DEFAULT_USER
            safe_user = user.replace("/", "_").replace("\\", "_").strip() or DEFAULT_USER
            file_path = self._directory / f"{safe_user}.log"
            line = self.format(record)
            with file_path.open("a", encoding=self._encoding) as handle:
                handle.write(line + "\n")
        except Exception:
            self.handleError(record)


_user_filter = UserContextFilter()
_formatter: Optional[logging.Formatter] = None
_configured = False


def configure_logging(force: bool = False) -> None:
    """Configure the root logger used across the Dash application."""
    global _configured, _formatter
    if _configured and not force:
        return

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    session_log = Path("log.log")
    session_log.write_text("", encoding="utf-8")
    archive_log = data_dir / "log-backup"
    archive_log.touch(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    _formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_formatter)
    stream_handler.addFilter(_user_filter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(session_log, encoding="utf-8")
    file_handler.setFormatter(_formatter)
    file_handler.addFilter(_user_filter)
    root_logger.addHandler(file_handler)

    archive_handler = logging.FileHandler(archive_log, encoding="utf-8")
    archive_handler.setFormatter(_formatter)
    archive_handler.addFilter(_user_filter)
    root_logger.addHandler(archive_handler)

    user_handler = UserFileHandler(data_dir)
    user_handler.setFormatter(_formatter)
    user_handler.addFilter(_user_filter)
    root_logger.addHandler(user_handler)

    _configured = True


def get_logger(name: Optional[str] = None) -> Logger:
    """Return a module-specific logger, configuring the system if needed."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


def set_user(user_id: Optional[str]) -> None:
    """Update the current user used for contextual logging."""
    _user_filter.set_user(user_id)


def current_user() -> str:
    """Return the user currently attached to log records."""
    return _user_filter.user


def _log(
    logger: Logger,
    level: int,
    message: str,
    args: Sequence[Any],
    project: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not _configured:
        configure_logging()

    extra_payload: Dict[str, Any] = dict(extra) if extra else {}
    extra_payload.setdefault("user", _user_filter.user)

    logger.log(level, message, *args, extra=extra_payload)

    if project is not None:
        if _formatter is None:
            formatted = f"{message % tuple(args) if args else message}"
        else:
            record = logger.makeRecord(
                logger.name,
                level,
                fn="",
                lno=0,
                msg=message,
                args=args,
                exc_info=None,
                extra=extra_payload,
            )
            formatted = _formatter.format(record)
        project.update_log(formatted + "\n")


def log_info(
    logger: Logger,
    message: str,
    *args: Any,
    project: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an informational message and update the optional project log."""
    _log(logger, logging.INFO, message, args, project=project, extra=extra)


def log_warning(
    logger: Logger,
    message: str,
    *args: Any,
    project: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a warning message and update the optional project log."""
    _log(logger, logging.WARNING, message, args, project=project, extra=extra)


def log_error(
    logger: Logger,
    message: str,
    *args: Any,
    project: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an error message and update the optional project log."""
    _log(logger, logging.ERROR, message, args, project=project, extra=extra)

