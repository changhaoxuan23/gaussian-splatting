from __future__ import annotations

from argparse import Action, ArgumentParser, Namespace
from json import loads as load_string
from typing import override

modifications = {}


class _ModificationAction(Action):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  @override
  def __call__(
    self,
    parser: ArgumentParser,
    namespace: Namespace,
    values: list[str],
    option_string: str | None = None,
  ) -> None:

    print("input", values)

    specifications = [(entry.split("."), value) for entry, value in (x.split("=") for x in values)]
    for entry, value in sorted(specifications, key=lambda x: len(x[0])):
      target = modifications
      for parent in entry[:-1]:
        if parent not in target:
          target[parent] = {}
        target = target[parent]
      target[entry[-1]] = load_string(value)


class _NoopAction(Action):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  @override
  def __call__(
    self,
    parser: ArgumentParser,
    namespace: Namespace,
    values: list[str],
    option_string: str | None = None,
  ) -> None: ...


def prepare_parser(parser: ArgumentParser) -> None:
  parser.add_argument("modifications", nargs="*", action=_ModificationAction)


def prepare_dummy_parser(parser: ArgumentParser) -> None:
  """Install a argument parser that does not really setup modifications.

  This parser just does nothing.
  """
  parser.add_argument("modifications", nargs="*", action=_NoopAction)
