"""Entry-point wrapper for scripts that touch BAR-ATH: tracebacks print type and shape info only, never values."""
from __future__ import annotations

import functools
import sys


def quiet(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            tb = e.__traceback__
            frames = []
            while tb:
                frames.append(f"{tb.tb_frame.f_code.co_filename.split('src')[-1]}:{tb.tb_lineno}")
                tb = tb.tb_next
            print(f"FAIL {type(e).__name__} at {' > '.join(frames[-3:])} (message withheld: may contain match data)")
            sys.exit(1)
    return wrapper


def verdict(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    return ok
