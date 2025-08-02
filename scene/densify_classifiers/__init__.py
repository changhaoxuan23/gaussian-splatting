"""Helper densify classifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch

from scene.densification_classifiers_typing import GaussianADCClassifier

if TYPE_CHECKING:
  from collections.abc import Callable, Iterable

  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class _LogicalMergingClassifier(GaussianADCClassifier):
  merger: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

  def __init__(
    self,
    classifiers: Iterable[GaussianADCClassifier, ...],
    *,
    allow_classifiers_visualize: bool = False,
  ) -> None:
    super().__init__()

    self._classifiers = classifiers
    self._allow_classifiers_visualize = allow_classifiers_visualize

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    selector = torch.zeros(size=(len(gaussians.get_xyz),), device=gaussians.get_xyz.device, dtype=bool)
    for classifier in self._classifiers:
      selector = self.merger(
        selector,
        classifier(
          gaussians=gaussians,
          visualizer=visualizer if self._allow_classifiers_visualize else None,
        ),
      )
    return selector


class IntersectionClassifier(_LogicalMergingClassifier):
  merger = torch.logical_and


class UnionClassifier(_LogicalMergingClassifier):
  merger = torch.logical_or
