"""Hooks used in training procedure of GaussianModels, when densifying the Gaussian."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
  import torch

  from scene.gaussian_protocols import TrainingGaussian


class NewGaussianPack(NamedTuple):
  point_id: torch.Tensor
  xyz: torch.Tensor
  features_dc: torch.Tensor
  features_rest: torch.Tensor
  opacity: torch.Tensor
  scaling: torch.Tensor
  rotation: torch.Tensor


class PendingDensifyDetails(NamedTuple):
  selector: torch.Tensor
  new_gaussian: NewGaussianPack


class PendingDensify(NamedTuple):
  clone: PendingDensifyDetails
  split: PendingDensifyDetails


class DensifyHook(ABC):
  @abstractmethod
  def before_densify_selection(self, gaussian: TrainingGaussian) -> None: ...

  @abstractmethod
  def after_each_densify_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None: ...

  @abstractmethod
  def before_densify_applied(
    self,
    gaussian: TrainingGaussian,
    points: PendingDensify,
  ) -> None: ...

  @abstractmethod
  def before_prune_selection(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...

  @abstractmethod
  def after_each_prune_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None: ...

  @abstractmethod
  def before_prune_applied(
    self,
    gaussian: TrainingGaussian,
    selector: torch.Tensor,
  ) -> None: ...

  @abstractmethod
  def before_done(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...

  @abstractmethod
  def after_train(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...


class HookManager:
  def __init__(self) -> None:
    self._hook_names = tuple(name for name, _ in inspect.getmembers(DensifyHook, inspect.isfunction))
    self._hooks: list[DensifyHook] = []
    for name in self._hook_names:

      def _invoke_hook(_name: str = name, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        for hook in self._hooks:
          getattr(hook, _name)(*args, **kwargs)

      setattr(self, name, _invoke_hook)

  def register_hook(self, hook: DensifyHook) -> None:
    self._hooks.append(hook)
