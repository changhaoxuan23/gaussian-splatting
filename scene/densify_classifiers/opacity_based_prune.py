"""Prune based on opacity."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from scene.densification_classifiers_typing import GaussianADCClassifier

if TYPE_CHECKING:
  import torch

  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class _OpacityBasedPruneClassifier(GaussianADCClassifier):
  def __init__(self, threshold: float) -> None:
    super().__init__()

    self._threshold = threshold

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    return (gaussians.get_opacity < self._threshold).squeeze()


Classifier = _OpacityBasedPruneClassifier
