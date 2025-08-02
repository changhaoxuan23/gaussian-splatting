"""Manager of extra metrics used in the training procedure of GaussianModels."""
from __future__ import annotations

from argparse import Namespace
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Self

import torch

from utils.modifications import modifications
from utils.projection_utils import to_3d_bbox
from utils.registers import registers

if TYPE_CHECKING:
  from collections.abc import Iterable

  from scene.cameras import Camera
  from scene.gaussian_protocols import RenderingGaussian


class MetricProjectionSpecification(NamedTuple):
  name: str
  lhs: torch.Tensor
  rhs: torch.Tensor | None
  denom: int | None

class ExtraMetricsManager:
  """Helper class holding extra metrics used for densify decisions."""

  class _MetricIntermediate:
    def __init__(self, depth: int | None = None) -> None:
      self._accumulation: torch.Tensor = torch.empty(0)
      self._denom: torch.Tensor = torch.empty(0)
      self._depth = depth

    @property
    def accumulation(self) -> torch.Tensor:
      return self._accumulation

    @accumulation.setter
    def accumulation(self, value: torch.Tensor) -> torch.Tensor:
      self._accumulation = value
      return self._accumulation

    @property
    def denom(self) -> torch.Tensor:
      return self._denom

    @denom.setter
    def denom(self, value: torch.Tensor) -> torch.Tensor:
      self._denom = value
      return self._denom

    def capture(self) -> tuple[torch.Tensor, torch.Tensor]:
      return self._accumulation, self._denom, self._depth

    def restore(self, state: tuple[torch.Tensor, torch.Tensor]) -> None:
      self._accumulation, self._denom, self._depth = state
    
    def reset(self, length: int, device: torch.device | str) -> None:
      _size = (length, ) if self._depth is None else (length, self._depth)
      self._accumulation = torch.zeros(size=_size, device=device)
      self._denom = torch.zeros(size=_size, device=device)

  def __init__(
    self,
    *,
    device: torch.device | str | None = "cuda",
  ) -> None:
    metrics = modifications.get("metrics-to-trace", ())

    if "old-projection" not in modifications:
      self._backward_helper = torch.empty(0)

    self._metrics: dict[str, ExtraMetricsManager._MetricIntermediate] = {}
    self._updated_metrics = set()

    def _access_helper(self: Self, name: str) -> torch.Tensor:
      if name not in self._updated_metrics:
        message = f"Accessing metric {name} which is never updated! "
        "Maybe you should add it into modifications.metrics-to-trace?"
        raise ValueError(message)
      temporary = self._metrics[name].accumulation / self._metrics[name].denom
      temporary[temporary.isnan()] = 0.0
      return temporary

    for metric in metrics:
      self._metrics[metric] = ExtraMetricsManager._MetricIntermediate()
      setattr(self.__class__, metric, property(fget=partial(_access_helper, name=metric)))

    self._device = device
    self._ordered_metrics_names = metrics
    self._render = registers.render

  def reset(self, *, points: int) -> None:
    """Reset all recorded metrics to zero and update number of points."""
    for metric in self._metrics:
      self._metrics[metric].reset(length=points, device=self._device)
    if "old-projection" not in modifications:
      self._backward_helper = torch.zeros(size=(points, 3), device=self._device, requires_grad=True)

  def filter_out(self, *, mask: torch.Tensor) -> None:
    """Remove records about points selected by the mask."""
    for record in self._metrics.values():
      record.accumulation = record.accumulation[~mask]
      record.denom = record.denom[~mask]
    if "old-projection" not in modifications:
      self._backward_helper = torch.zeros(size=((~mask).sum(), 3), device=self._device, requires_grad=True)

  def project_old(  # noqa: PLR0913
    self,
    *,
    position: torch.Tensor,
    boxes: tuple[torch.Tensor, torch.Tensor],
    segments: tuple[torch.Tensor, torch.Tensor],
    cameras: tuple[Camera, Camera],
    metrics: tuple[MetricProjectionSpecification, ...],
    lhs_only: bool,
  ) -> None:
    if "old-projection" not in modifications or any(metric.name not in self._metrics for metric in metrics):
      raise ValueError
    _convert_result = to_3d_bbox(boxes=boxes, cameras=cameras)
    if _convert_result is None:
      return
    (x_min_3d, x_max_3d), (y_min_3d, y_max_3d), (z_min_3d, z_max_3d) = _convert_result

    # select gaussians in the intersected space
    mask_x = torch.logical_and(position[:, 0] > x_min_3d, position[:, 0] < x_max_3d)
    mask_y = torch.logical_and(position[:, 1] > y_min_3d, position[:, 1] < y_max_3d)
    mask_z = torch.logical_and(position[:, 2] > z_min_3d, position[:, 2] < z_max_3d)
    mask_xy = torch.logical_and(mask_x, mask_y)
    mask = torch.logical_and(mask_xy, mask_z)

    for metric in metrics:
      name, lhs, rhs = metric.name, metric.lhs, metric.rhs
      source = lhs[segments[0]] if lhs_only else torch.cat((lhs[segments[0]], rhs[segments[1]]))
      self._metrics[name].accumulation[mask] += source.mean()
      self._metrics[name].denom[mask] += metric.denom if metric.denom is not None else 1
      self._updated_metrics.add(name)

  def project(
    self,
    *,
    gaussian: RenderingGaussian,
    camera: Camera,
    metrics: tuple[MetricProjectionSpecification, ...],
  ) -> None:
    if "old-projection" in modifications or any(metric.name not in self._metrics for metric in metrics):
      raise ValueError
    pipe_configuration = Namespace(debug=False, antialiasing=True, compute_cov3D_python=False)
    render_result = self._render(
      viewpoint_camera=camera,
      pc=gaussian,
      pipe=pipe_configuration,
      bg_color=torch.zeros(size=(3,), device="cuda"),
      override_color=self._backward_helper.view(-1, 3),
    )
    factors, visibility_filter, image_plane_radius = (
      render_result["render"],
      render_result["visibility_filter"],
      render_result["radii"],
    )
    update_filter = visibility_filter.squeeze(-1)
    for metric in metrics:
      name, value, denom = metric.name, metric.lhs, metric.denom
      if denom == 0:
        continue
      helper = (factors[0, ...] * value.detach()).sum()
      (projection_result,) = torch.autograd.grad(
        outputs=helper,
        inputs=self._backward_helper,
        retain_graph=True,
        create_graph=False,
      )

      # align the scale to per-pixel
      projection_result[update_filter] /= image_plane_radius[update_filter, None] ** 2

      self._metrics[name].accumulation[update_filter] += projection_result[update_filter, 0]
      self._metrics[name].denom[update_filter] += denom if denom is not None else 1
      self._updated_metrics.add(name)

  def capture(self) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple((name, getattr(self, name)) for name in self._ordered_metrics_names)

  def capture_state(self) -> tuple[tuple[str, tuple[torch.Tensor, torch.Tensor]], ...]:
    return tuple(
      (
        name,
        self._metrics[name].capture(),
      )
      for name in self._ordered_metrics_names
    )

  def restore_state(self, state: tuple[tuple[str, tuple[torch.Tensor, torch.Tensor]], ...]) -> tuple:
    for name, (captured_name, inner_state) in zip(self._ordered_metrics_names, state, strict=True):
      if name != captured_name:
        message = f"name mismatched: captured {captured_name}, but expect {name}"
        raise ValueError(message)
      self._metrics[name].restore(state=inner_state)
