"""Hook that dumps attributes of gaussian."""

from shutil import rmtree
from typing import override

import torch

from scene.densification_hook_typing import DensifyHook, PendingDensify
from scene.gaussian_protocols import TrainingGaussian
from utils.modifications import modifications
from utils.registers import registers


class AttributeDumper(DensifyHook):
  def __init__(self) -> None:
    self._output_directory = registers.out_path.joinpath("attributes_dump")
    rmtree(path=self._output_directory, ignore_errors=True)
    self._output_directory.mkdir(exist_ok=False)

  @override
  def before_densify_selection(self, gaussian: TrainingGaussian) -> None:
    _scaling = gaussian.get_scaling
    volume = _scaling[:, 0] * _scaling[:, 1] * _scaling[:, 2] * torch.pi * 4 / 3
    torch.save(
      {
        "opacity": gaussian.get_opacity[..., 0].cpu(),
        "volume": volume.cpu(),
      },
      self._output_directory.joinpath(f"{registers.train_progress.iteration:05d}"),
    )

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
  ) -> None: ...

  @override
  def after_train(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...


def _should_launch() -> bool:
  return modifications.get("dump-attribute-distribution", False)


HOOK = AttributeDumper
HOOK_REGISTER_CONDITION = _should_launch
