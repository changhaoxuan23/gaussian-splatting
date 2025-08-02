"""Hooks used in densification."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  from scene.densification_hook_typing import HookManager


class HookEquipped(Protocol):
  densify_hook_manager: HookManager


def install_hooks(target: HookEquipped) -> None:
  hook_directory = Path(__file__).parent.joinpath("densify_hooks")
  if not hook_directory.is_dir():
    return
  for candidate_path in hook_directory.iterdir():
    if not candidate_path.is_file():
      continue
    if candidate_path.suffix != ".py":
      continue
    if candidate_path.name == "__init__.py":
      continue
    spec = spec_from_file_location("", candidate_path)
    hook_module = module_from_spec(spec)
    spec.loader.exec_module(hook_module)
    if hook_module.HOOK_REGISTER_CONDITION():
      target.densify_hook_manager.register_hook(hook=hook_module.HOOK())
      print(f"installed hook: {hook_module.HOOK.__name__}")
