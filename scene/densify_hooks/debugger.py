"""Hook that dumps statistics in ADC."""

from typing import override

import torch

from scene.densification_hook_typing import DensifyHook, PendingDensify
from scene.gaussian_protocols import TrainingGaussian


class Debugger(DensifyHook):
  def __init__(self) -> None: ...

  @override
  def before_densify_selection(self, gaussian: TrainingGaussian) -> None: ...

  @override
  def after_each_densify_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None: ...

  @override
  def before_densify_applied(
    self,
    gaussian: TrainingGaussian,
    points: PendingDensify,
  ) -> None: ...

  @override
  def before_prune_selection(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...

  @override
  def after_each_prune_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None: ...

  @override
  def before_prune_applied(
    self,
    gaussian: TrainingGaussian,
    selector: torch.Tensor,
  ) -> None: ...

  @override
  def before_done(
    self,
    gaussian: TrainingGaussian,
  ) -> None:
    print(f"gaussians: {len(gaussian.get_xyz)}")

  @override
  def after_train(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...


def should_launch() -> bool:
  return False


HOOK = Debugger
HOOK_REGISTER_CONDITION = should_launch
