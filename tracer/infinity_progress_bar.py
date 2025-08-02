from __future__ import annotations

from hashlib import blake2b, shake_256
from secrets import token_bytes
from sys import stdout
from termios import tcgetwinsize
from threading import Thread
from time import sleep
from weakref import ref


def _get_terminal_width() -> int:
  return tcgetwinsize(stdout.fileno())[1]


def _render_helper(target: ref) -> None:
  while True:
    _target = target()
    if _target is None:
      break
    _target.render()
    sleep(0.1)


class InfinityProgressBar:
  def __init__(self) -> None:
    self._prefix_source = 0
    self._prefix_generator_base = blake2b(
      key=token_bytes(64),
      salt=token_bytes(16),
      person=token_bytes(16),
    )
    self._prefix_final_base = shake_256()
    self._information: dict[str, str] = {}

    Thread(target=_render_helper, kwargs={"target": ref(self)}, daemon=True).start()

  def __del__(self) -> None:
    print(file=stdout)

  def config_information(self, name: str, value: object | None) -> None:
    if value is None:
      if name in self._information:
        del self._information[name]
      return

    self._information[name] = str(value)

  def render(self) -> None:
    _suffix = ", ".join(f"{name}: {value}" for name, value in self._information.items())
    suffix = f" [{_suffix}]"
    prefix_generator = self._prefix_generator_base.copy()
    prefix_generator.update(str(self._prefix_source).encode("ascii"))
    prefix_postprocessor = self._prefix_final_base.copy()
    prefix_postprocessor.update(prefix_generator.digest())
    _length = _get_terminal_width() - len(suffix)
    prefix = prefix_postprocessor.hexdigest(_length // 2 + 1)[:_length]
    prefix = prefix.replace("0d000721", "\x1b[1;4;34m0d000721\x1b[0m")
    print(f"\r{prefix}{suffix}", file=stdout, end="")
    self._prefix_source += 1
