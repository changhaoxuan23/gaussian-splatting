"""Densify based on window matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, override

import torch

from scene.densification_classifiers_typing import GaussianADCClassifier
from utils.projection_utils import to_3d_bbox

if TYPE_CHECKING:
  from scene.cameras import Camera
  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer


class WindowSources(TypedDict):
  box_pairs: list[tuple[tuple[int, torch.Tensor], tuple[int, torch.Tensor]]]
  cameras: list[Camera]


def _prepare_boxes(
  box_pairs: list[tuple[tuple[int, torch.Tensor], tuple[int, torch.Tensor]]],
  cameras: list[Camera],
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
  return [
    box3d
    for box3d in [
      to_3d_bbox(
        boxes=(lhs_bbox, rhs_bbox),
        cameras=(cameras[lhs_camera_id], cameras[rhs_camera_id]),
      )
      for (lhs_camera_id, lhs_bbox), (rhs_camera_id, rhs_bbox) in box_pairs
    ]
    if box3d is not None
  ]


class _WindowBasedDensifyClassifier(GaussianADCClassifier):
  def __init__(self, source: WindowSources) -> None:
    super().__init__()

    self._source = source

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    boxes = _prepare_boxes(box_pairs=self._source.box_pairs, cameras=self._source.cameras)
    xyz = gaussians.get_xyz
    selector = torch.zeros(size=(len(xyz),), device=xyz.device, dtype=bool)
    for (x_min_3d, x_max_3d), (y_min_3d, y_max_3d), (z_min_3d, z_max_3d) in boxes:
      mask_x = torch.logical_and(xyz[:, 0] > x_min_3d, xyz[:, 0] < x_max_3d)
      mask_y = torch.logical_and(xyz[:, 1] > y_min_3d, xyz[:, 1] < y_max_3d)
      mask_z = torch.logical_and(xyz[:, 2] > z_min_3d, xyz[:, 2] < z_max_3d)
      mask_xyz = torch.logical_and(mask_x, mask_y)
      mask_xyz = torch.logical_and(mask_xyz, mask_z)
      selector = selector.logical_or(mask_xyz)

    return selector


Classifier = _WindowBasedDensifyClassifier
