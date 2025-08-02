"""Densify based on grad norm."""

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING, override

import torch

from gaussian_renderer import render
from scene.densification_classifiers_typing import GaussianADCClassifier
from utils.modifications import modifications
from utils.visualize_utils import clamp_extreme_values

if TYPE_CHECKING:
  from arguments import OptimizationParameters
  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class _GradBasedDensifyClassifier(GaussianADCClassifier):
  def __init__(self, optimization_config: OptimizationParameters) -> None:
    super().__init__()

    self._optimization_config = optimization_config

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    threshold = self._optimization_config.densify_grad_threshold
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads[grads.isnan()] = 0.0
    # apply modifier if running the multiplier variant
    if "multiplier" in modifications:
      grads *= gaussians.extra_metrics.ldl
    # select by grad threshold
    grad_norm = torch.norm(grads, dim=-1)

    if visualizer is not None:
      _pipe_configuration = Namespace(
        debug=False,
        antialiasing=True,
        compute_cov3D_python=False,
        convert_SHs_python=False,
      )
      _device = gaussians.get_xyz.device
      _zero_level_color = torch.zeros(size=(3,), device=_device)
      _positive_color = torch.as_tensor((255, 87, 34), device=_device) / 255
      _negative_color = torch.as_tensor((0, 188, 212), device=_device) / 255

      # calculate color overrides
      _grad_norm = clamp_extreme_values(data=grad_norm, minimum=0.05, maximum=0.05, cutting_point=threshold)
      color_override = torch.empty(size=(len(gaussians.get_xyz), 3), device=_device)
      color_override[_grad_norm == threshold] = _zero_level_color
      color_override[_grad_norm > threshold] = (
        (_grad_norm[_grad_norm > threshold, None] - threshold)
        / (_grad_norm.max() - threshold)
        * _negative_color
      )
      color_override[_grad_norm < threshold] = (
        (threshold - _grad_norm[_grad_norm < threshold, None])
        / (threshold - _grad_norm.min())
        * _positive_color
      )
      for adc_view in visualizer.views:
        _render_result = render(
          viewpoint_camera=adc_view.view,
          pc=gaussians,
          pipe=_pipe_configuration,
          bg_color=torch.zeros(size=(3,), device="cuda"),
          override_color=color_override,
        )["render"]
        adc_view.register_visualization(
          annotation=(
            "position gradient",
            f"L={_grad_norm.min().item()}",
            f"RL={grad_norm.min().item()}",
            f"H={_grad_norm.max().item()}",
            f"RH={grad_norm.max().item()}",
          ),
          image=_render_result,
        )

    return grad_norm >= threshold


Classifier = _GradBasedDensifyClassifier
