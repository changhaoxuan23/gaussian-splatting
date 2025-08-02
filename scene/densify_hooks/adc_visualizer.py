"""Hook that visualizes the ADC procedure."""

from argparse import Namespace
from shutil import rmtree
from typing import override

import torch

from scene.densification_hook_typing import DensifyHook, PendingDensify
from scene.gaussian_protocols import TrainingGaussian
from utils.adc_visualizer import ADCVisualizer, register_gaussian_counting
from utils.loss_utils import ssim
from utils.modifications import modifications
from utils.registers import registers
from utils.visualize_utils import visualize_gaussian_count


class ADCVisualizerHook(DensifyHook):
  def __init__(self) -> None:
    _views = (
      3
      if isinstance(modifications["visualize-adc"], bool)
      else modifications["visualize-adc"].get("views", 3)
    )
    self._visualizer = ADCVisualizer(cameras=registers.training_cameras, views=_views, seed="ADC visualizer")
    self._should_initialize = True
    self._render_configuration = Namespace(
      debug=False,
      antialiasing=True,
      compute_cov3D_python=False,
      convert_SHs_python=False,
    )
    self._visualize_output_directory = registers.out_path.joinpath("adc_visualize")
    rmtree(path=self._visualize_output_directory, ignore_errors=True)
    self._visualize_output_directory.mkdir(exist_ok=False)

  def _render_component_visualization(
    self,
    prefix: str,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None:
    for adc_view in self._visualizer.views:
      counting_result = visualize_gaussian_count(
        gaussian=gaussian,
        camera=adc_view.view,
        selector=selector,
      )
      register_gaussian_counting(
        adc_visualize_view=adc_view,
        counting_result=counting_result,
        annotation_prefixes=(f"{prefix} from {name}",),
      )

  @override
  def before_densify_selection(self, gaussian: TrainingGaussian) -> None:
    if self._should_initialize:
      gaussian.adc_visualizer = self._visualizer
      self._should_initialize = False
    if isinstance(modifications["visualize-adc"], bool) or not modifications["visualize-adc"].get(
      "with-reference",
      False,
    ):
      return

    for _adc_view in self._visualizer.views:
      _view, _original_name = _adc_view.view, _adc_view.name
      _render_result = registers.render(
        viewpoint_camera=_view,
        pc=gaussian,
        pipe=self._render_configuration,
        bg_color=torch.zeros(size=(3,), device="cuda"),
      )["render"]
      _l1_loss_image = (
        torch.abs(_render_result - _view.original_image).mean(dim=0, keepdims=True).expand(3, -1, -1)
      )
      _ssim_loss_image = 1 - (
        ssim(_render_result, _view.original_image, with_map=True)[1]
        .mean(dim=0, keepdims=True)
        .expand(3, -1, -1)
      )
      _adc_view.register_visualization(annotation=(_original_name,), image=_render_result)
      _adc_view.register_visualization(annotation=("l1",), image=_l1_loss_image)
      _adc_view.register_visualization(annotation=("ssim",), image=_ssim_loss_image)

  @override
  def after_each_densify_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None:
    self._render_component_visualization(
      prefix="densify",
      gaussian=gaussian,
      name=name,
      selector=selector,
    )

  @override
  def before_densify_applied(
    self,
    gaussian: TrainingGaussian,
    points: PendingDensify,
  ) -> None:
    for _adc_view in self._visualizer.views:
      register_gaussian_counting(
        adc_visualize_view=_adc_view,
        counting_result=visualize_gaussian_count(
          gaussian=gaussian,
          camera=_adc_view.view,
          selector=points.clone.selector.logical_or(points.split.selector),
        ),
        annotation_prefixes=("total densify",),
      )
      register_gaussian_counting(
        adc_visualize_view=_adc_view,
        counting_result=visualize_gaussian_count(
          gaussian=gaussian,
          camera=_adc_view.view,
          selector=points.clone.selector,
        ),
        annotation_prefixes=("densify by clone",),
      )
      register_gaussian_counting(
        adc_visualize_view=_adc_view,
        counting_result=visualize_gaussian_count(
          gaussian=gaussian,
          camera=_adc_view.view,
          selector=points.split.selector,
        ),
        annotation_prefixes=("densify by split",),
      )

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
  ) -> None:
    self._render_component_visualization(
      prefix="prune",
      gaussian=gaussian,
      name=name,
      selector=selector,
    )

  @override
  def before_prune_applied(
    self,
    gaussian: TrainingGaussian,
    selector: torch.Tensor,
  ) -> None:
    for _adc_view in self._visualizer.views:
      register_gaussian_counting(
        adc_visualize_view=_adc_view,
        counting_result=visualize_gaussian_count(
          gaussian=gaussian,
          camera=_adc_view.view,
          selector=selector,
        ),
        annotation_prefixes=("total prune",),
      )

  @override
  def before_done(
    self,
    gaussian: TrainingGaussian,
  ) -> None:
    _output_path = self._visualize_output_directory.joinpath(f"{registers.train_progress.iteration:05d}.png")
    self._visualizer.save(destination=_output_path)

  @override
  def after_train(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...


def should_launch() -> bool:
  return modifications.get("visualize-adc", False)


HOOK = ADCVisualizerHook
HOOK_REGISTER_CONDITION = should_launch
