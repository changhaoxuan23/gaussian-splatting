"""Typing helpers for densification classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import torch

  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class GaussianADCClassifier(ABC):
  @abstractmethod
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor: ...
