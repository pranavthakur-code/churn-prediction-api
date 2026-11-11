"""One JSON object per log line on stdout, ready for any log shipper."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "churn_api"
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": datetime.fromtimestamp(record.created, timezone.utc)
               .isoformat(timespec="milliseconds"),
               "level": record.levelname, "logger": record.name,
               "event": record.getMessage()}
        out.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = True            # lets pytest's caplog see records
    if not any(getattr(h, "_churn", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        h._churn = True
        logger.addHandler(h)
    return logger
