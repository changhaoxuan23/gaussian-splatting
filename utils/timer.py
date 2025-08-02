"""A simple timer that works like bash builtin -- time."""
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import monotonic_ns


class Timer:
  def __init__(self) -> None:
    self._start_time = monotonic_ns()
    self._start_usage = getrusage(RUSAGE_SELF)
  def save(self, target: Path) -> None:
    _end_usage = getrusage(RUSAGE_SELF)
    _end_time = monotonic_ns()
    target.write_text(
      f"real: {_end_time - self._start_time}ns\n"
      f"user: {_end_usage.ru_utime - self._start_usage.ru_utime}s\n"
      f"sys : {_end_usage.ru_stime - self._start_usage.ru_stime}s\n",
    )
