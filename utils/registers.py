"""Global dictionary for dirty and quick information transmission."""
from io import StringIO
from sys import stderr
from traceback import print_stack
from typing import Any

from utils.modifications import modifications


class _Registers:
  def __init__(self) -> None:
    self._map = {}

  def register(self, name: str, value: Any) -> None:  # noqa: ANN401
    if "debug" in modifications:
      buffer = StringIO("\n")
      buffer.write("=" * 80)
      buffer.write(f"\n New entry <{name}> registered via following invocations:\n")
      print_stack(file=buffer)
      buffer.write("\n")
      buffer.write("=" * 80)
      buffer.write("\n")

      print(buffer.getvalue(), file=stderr)

    self._map[name] = value
    object.__setattr__(self, name, self._map[name])

  def register_once(self, name: str, value: Any) -> bool:  # noqa: ANN401
    if name in self._map:
      return False
    self.register(name=name, value=value)
    return True

registers = _Registers()
