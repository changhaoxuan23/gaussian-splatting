"""Protocol definition of GaussianModels."""
from typing import Protocol

import torch

from arguments import OptimizationParameters
from scene.extra_metrics_manager import ExtraMetricsManager


class RenderingGaussian(Protocol):
  """Interface of GaussianModels ready for rendering."""

  active_sh_degree: int

  @property
  def get_scaling(self) -> torch.Tensor: ...

  @property
  def get_rotation(self) -> torch.Tensor: ...

  @property
  def get_xyz(self) -> torch.Tensor: ...

  @property
  def get_features(self) -> torch.Tensor: ...

  @property
  def get_features_dc(self) -> torch.Tensor: ...

  @property
  def get_features_rest(self) -> torch.Tensor: ...

  @property
  def get_opacity(self) -> torch.Tensor: ...

  def get_covariance(self, scaling_modifier: float) -> torch.Tensor: ...

class TrainingGaussian(RenderingGaussian, Protocol):
  """Interface for GaussianModels being trained."""

  xyz_gradient_accum: torch.Tensor
  denom: torch.Tensor
  extra_metrics: ExtraMetricsManager
  registers: dict
  point_id: torch.Tensor

  def capture_trace(self) -> tuple[torch.Tensor]: ...
  def capture(self) -> tuple: ...
  def restore(self, model_args: tuple, training_args: OptimizationParameters) -> None: ...
