"""Densify based on ldl (or "metric")."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from scene.densification_classifiers_typing import GaussianADCClassifier

if TYPE_CHECKING:
  import torch

  from arguments import OptimizationParameters
  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class _LdlBasedDensifyClassifier(GaussianADCClassifier):
  def __init__(self, optimization_config: OptimizationParameters) -> None:
    super().__init__()

    self._optimization_config = optimization_config

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    threshold = self._optimization_config.densify_metric_threshold
    return gaussians.extra_metrics.ldl >= threshold


Classifier = _LdlBasedDensifyClassifier
