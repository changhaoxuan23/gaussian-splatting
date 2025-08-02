"""Hook that dumps statistics in ADC."""

import json
from typing import override

import torch

from scene.densification_hook_typing import DensifyHook, PendingDensify
from scene.gaussian_protocols import TrainingGaussian
from utils.modifications import modifications
from utils.registers import registers


class ADCStatisticsDumper(DensifyHook):
  def __init__(self) -> None:
    self._target_path = registers.out_path.joinpath("adc_statistics")
    self._target_path.unlink(missing_ok=True)

  def _add_component(self, target: str, name: str, selector: torch.Tensor) -> None:
    self._statistics[target][name] = selector.sum().item()

  @override
  def before_densify_selection(self, gaussian: TrainingGaussian) -> None:
    self._statistics = {
      "iteration": registers.train_progress.iteration,
      "before_ADC": len(gaussian.get_xyz),
      "densify_components": {},
      "prune_components": {},
    }

  @override
  def after_each_densify_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None:
    self._add_component(target="densify_components", name=name, selector=selector)

  @override
  def before_densify_applied(
    self,
    gaussian: TrainingGaussian,
    points: PendingDensify,
  ) -> None:
    self._statistics["densify"] = (points.clone.selector.sum() + points.split.selector.sum()).item()
    self._statistics["densify_clone"] = points.clone.selector.sum().item()
    self._statistics["densify_split"] = points.split.selector.sum().item()

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
  ) -> None:
    self._add_component(target="prune_components", name=name, selector=selector)

  @override
  def before_prune_applied(
    self,
    gaussian: TrainingGaussian,
    selector: torch.Tensor,
  ) -> None:
    self._statistics["pruned"] = selector.clone.sum().item()

  @override
  def before_done(
    self,
    gaussian: TrainingGaussian,
  ) -> None:
    self._statistics["after_ADC"] = len(gaussian.get_xyz)
    with self._target_path.open("a") as f:
      json.dump(self._statistics, f)
      f.write("\n")

  @override
  def after_train(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...


def should_launch() -> bool:
  return modifications.get("dump_ADC_statistics", False)


HOOK = ADCStatisticsDumper
HOOK_REGISTER_CONDITION = should_launch
