"""Hook that validates the correspondence of metrics and ADC selections."""

from argparse import Namespace
from shutil import rmtree
from typing import override

import torch

from scene.densification_hook_typing import DensifyHook, PendingDensify
from scene.gaussian_protocols import TrainingGaussian
from tracer.evaluation_helper import calculate_metric_difference
from utils.adc_visualizer import ADCVisualizer
from utils.modifications import modifications
from utils.registers import registers
from utils.visualize_utils import MaskedGaussianModelView, clamp_extreme_values


class _ClonedGaussian:
  def __init__(self, source: TrainingGaussian) -> None:
    self._xyz = source.get_xyz.clone()
    self._scaling = source.get_scaling.clone()
    self._rotation = source.get_rotation.clone()
    self._opacity = source.get_opacity.clone()

    self.active_sh_degree = source.active_sh_degree
    self.point_id = source.point_id.clone()

  @property
  def get_scaling(self) -> torch.Tensor:
    return self._scaling

  @property
  def get_rotation(self) -> torch.Tensor:
    return self._rotation

  @property
  def get_xyz(self) -> torch.Tensor:
    return self._xyz

  @property
  def get_features(self) -> torch.Tensor:
    raise NotImplementedError

  @property
  def get_features_dc(self) -> torch.Tensor:
    raise NotImplementedError

  @property
  def get_features_rest(self) -> torch.Tensor:
    raise NotImplementedError

  @property
  def get_opacity(self) -> torch.Tensor:
    return self._opacity

  def get_covariance(self, scaling_modifier: float) -> torch.Tensor:
    raise NotImplementedError


class TraceValidator(DensifyHook):
  def __init__(self) -> None:
    self._output_directory = registers.out_path.joinpath("trace_validation")
    rmtree(path=self._output_directory, ignore_errors=True)
    self._output_directory.mkdir(exist_ok=False)

    self._metric_names = "ldl", "l1", "ssim"

    self._saved_gaussian: _ClonedGaussian | None = None

    _views = (
      3
      if isinstance(modifications["visualize-adc"], bool)
      else modifications["visualize-adc"].get("views", 3)
    )
    self._visualizer = ADCVisualizer(cameras=registers.training_cameras, views=_views, seed="ADC visualizer")
    self._pipe_configuration = Namespace(
      debug=False,
      antialiasing=True,
      compute_cov3D_python=False,
      convert_SHs_python=False,
    )

  @torch.no_grad()
  def _get_current_metrics(self, gaussian: TrainingGaussian) -> torch.Tensor:
    return torch.stack(
      (
        gaussian.extra_metrics.ldl,
        gaussian.extra_metrics.l1,
        gaussian.extra_metrics.ssim,
      ),
      dim=-1,
    )

  def _validate(self, saved_gaussian: _ClonedGaussian, current_gaussian: TrainingGaussian) -> None:
    current_metrics = self._get_current_metrics(gaussian=current_gaussian)
    split_result = calculate_metric_difference(
      source_points=self._saved_selection,
      source_metrics=self._saved_metrics,
      mapping=self._saved_split_mapping,
      current_points=current_gaussian.point_id,
      current_metrics=current_metrics,
    )
    clone_result = calculate_metric_difference(
      source_points=self._saved_selection,
      source_metrics=self._saved_metrics,
      mapping=self._saved_clone_mapping,
      current_points=current_gaussian.point_id,
      current_metrics=current_metrics,
    )
    _device = saved_gaussian.get_xyz.device
    _zero_level_color = torch.zeros(size=(3,), device=_device)
    _positive_color = torch.as_tensor((255, 87, 34), device=_device) / 255
    _negative_color = torch.as_tensor((0, 188, 212), device=_device) / 255

    selector = torch.cat(
      (
        torch.where(self._saved_selector)[0][split_result.effective_source_points],
        torch.where(self._saved_selector)[0][clone_result.effective_source_points],
      ),
    )
    rendering_gaussian = MaskedGaussianModelView(gaussian=saved_gaussian, mask=selector)

    for i in range(split_result.absolute_difference.shape[-1]):
      color_override = torch.zeros(size=(len(rendering_gaussian.get_xyz), 3), device=_device)
      _difference = torch.cat(
        (
          split_result.absolute_difference[:, i],
          clone_result.absolute_difference[:, i],
        ),
        dim=0,
      )
      difference = clamp_extreme_values(data=_difference, minimum=0.01, maximum=0.01, cutting_point=0)
      for adc_view in self._visualizer.views:
        # calculate_metric_difference reports source_metrics - current_metrics, therefore positive difference
        #  indicates decreased metrics (losses) and negative difference indicates increased metrics.
        # mark positive values with blue and negative values with red
        color_override[difference > 0] = difference[difference > 0, None] / difference.max() * _negative_color
        color_override[difference < 0] = difference[difference < 0, None] / difference.min() * _positive_color
        _render_result = registers.render(
          viewpoint_camera=adc_view.view,
          pc=rendering_gaussian,
          pipe=self._pipe_configuration,
          bg_color=torch.zeros(size=(3,), device="cuda"),
          override_color=color_override,
        )["render"]
        adc_view.register_visualization(
          annotation=(
            f"d{self._metric_names[i]}",
            f"L={difference.min().item()}",
            f"RL={_difference.min().item()}",
            f"H={difference.max().item()}",
            f"RH={_difference.max().item()}",
          ),
          image=_render_result,
        )

        # make also a binary version
        color_override[difference > 0] = _negative_color
        color_override[difference < 0] = _positive_color
        _render_result = registers.render(
          viewpoint_camera=adc_view.view,
          pc=rendering_gaussian,
          pipe=self._pipe_configuration,
          bg_color=torch.zeros(size=(3,), device="cuda"),
          override_color=color_override,
        )["render"]
        adc_view.register_visualization(
          annotation=(f"d{self._metric_names[i]}(binary)",),
          image=_render_result,
        )
    self._visualizer.save(
      destination=self._output_directory.joinpath(f"{registers.train_progress.iteration:05d}.png"),
    )

  @override
  def before_densify_selection(self, gaussian: TrainingGaussian) -> None:
    if self._saved_gaussian is not None:
      self._validate(saved_gaussian=self._saved_gaussian, current_gaussian=gaussian)

    self._saved_gaussian = _ClonedGaussian(source=gaussian)

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
  ) -> None:
    self._saved_clone_mapping = torch.cat(
      (
        gaussian.point_id[points.clone.selector][:, None],
        gaussian.point_id[points.clone.selector][:, None],
        points.clone.new_gaussian.point_id[:, None],
      ),
      dim=-1,
    )
    self._saved_split_mapping = torch.cat(
      (
        gaussian.point_id[points.split.selector][:, None],
        points.split.new_gaussian.point_id.view(points.split.selector.sum(), -1),
      ),
      dim=-1,
    )
    self._saved_selector = points.clone.selector.logical_or(points.split.selector)
    self._saved_selection = gaussian.point_id[self._saved_selector].clone()
    self._saved_metrics = self._get_current_metrics(gaussian=gaussian)[self._saved_selector].clone()

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
  ) -> None:
    if self._saved_gaussian is not None:
      self._validate(saved_gaussian=self._saved_gaussian, current_gaussian=gaussian)


def _should_launch() -> bool:
  return "trace" in modifications and modifications["trace"].get("validate-capture", False)


HOOK = TraceValidator
HOOK_REGISTER_CONDITION = _should_launch
