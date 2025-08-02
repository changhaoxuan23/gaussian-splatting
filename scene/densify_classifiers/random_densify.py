"""Densify randomly."""

from __future__ import annotations

from random import sample
from typing import TYPE_CHECKING, override

import torch

from scene.densification_classifiers_typing import GaussianADCClassifier

if TYPE_CHECKING:
  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class _RandomDensifyClassifier(GaussianADCClassifier):
  def __init__(self, ratio: float) -> None:
    super().__init__()

    self._ratio = ratio

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    n_gaussians = len(gaussians.get_xyz)
    randomly_pickup_count = int(n_gaussians * self._ratio)
    selector = torch.zeros(size=(n_gaussians,), device=gaussians.get_xyz.device, dtype=bool)
    selector[sample(range(n_gaussians), k=randomly_pickup_count)] = True
    return selector


Classifier = _RandomDensifyClassifier
