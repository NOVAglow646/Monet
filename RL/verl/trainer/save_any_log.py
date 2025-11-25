# tee_logger.py
import sys
import os
import datetime
import logging
import threading
import faulthandler
from types import TracebackType
from typing import TextIO, Optional, Type

def setup_tee_logger(log_path: str) -> None:
    import sys, os, datetime, logging, threading, faulthandler
    from types import TracebackType
    from typing import Optional, Type

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
            self._lock = threading.Lock()

        def write(self, data):
            with self._lock:
                for s in self._streams:
                    s.write(data)
                    s.flush()

        def flush(self):
            with self._lock:
                for s in self._streams:
                    s.flush()

        def fileno(self):  # <- 新增
            for s in self._streams:
                if hasattr(s, "fileno"):
                    return s.fileno()
            raise AttributeError("No underlying stream provides fileno()")

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    # 替换前，先启用 faulthandler（也可以放后面，任取其一）
    faulthandler.enable(file=sys.__stderr__, all_threads=True)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    def _excepthook(exc_type: Type[BaseException],
                    exc_value: BaseException,
                    exc_tb: Optional[TracebackType]) -> None:
        logging.critical("Uncaught exception",
                         exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _excepthook
    logging.info("Tee logger initialized -> %s", log_path)

