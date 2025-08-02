"""Prune based on scale."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch

from scene.densification_classifiers_typing import GaussianADCClassifier

if TYPE_CHECKING:
  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class _ScaleBasedPruneClassifier(GaussianADCClassifier):
  def __init__(self, max_screen_size: float, extent: float) -> None:
    super().__init__()

    self._max_screen_size = max_screen_size
    self._extent = extent

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    return torch.logical_or(
      gaussians.max_radii2D > self._max_screen_size,
      gaussians.get_scaling.max(dim=1).values > 0.1 * self._extent,
    )


Classifier = _ScaleBasedPruneClassifier
