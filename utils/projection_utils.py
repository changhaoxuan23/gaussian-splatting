"""Utilities for 2d-3d projections."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from scene.cameras import Camera


_epsilon = 1e-6

def _intersect_lines(
  ray1_origin: torch.Tensor,
  ray1_dir: torch.Tensor,
  ray2_origin: torch.Tensor,
  ray2_dir: torch.Tensor,
) -> torch.Tensor:
  # Normalize direction vectors
  ray1_dir = ray1_dir / torch.norm(ray1_dir)
  ray2_dir = ray2_dir / torch.norm(ray2_dir)

  # Cross product of direction vectors
  cross_dir = torch.cross(ray1_dir, ray2_dir)
  print(ray1_dir.shape, ray2_dir.shape)
  cross_dir_norm = torch.norm(cross_dir)

  # Check if the rays are parallel
  if cross_dir_norm < _epsilon:
    return None  # Rays are parallel and do not intersect

  # Line between the origins
  origin_diff = ray2_origin - ray1_origin

  # Calculate the distance along the cross product direction
  t1 = torch.dot(torch.cross(origin_diff, ray2_dir), cross_dir) / (cross_dir_norm**2)
  t2 = torch.dot(torch.cross(origin_diff, ray1_dir), cross_dir) / (cross_dir_norm**2)

  # Closest points on each ray
  closest_point1 = ray1_origin + t1 * ray1_dir
  closest_point2 = ray2_origin + t2 * ray2_dir

  # Midpoint between the two closest points as the intersection point
  return (closest_point1 + closest_point2) / 2.0


def to_3d_bbox(
  boxes: tuple[torch.Tensor, torch.Tensor],
  cameras: tuple[Camera, Camera],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
  """Return ((x_low, x_high), (y_low, y_high), (z_low, z_high))."""
  box0, box1 = (box.cpu() for box in boxes)
  if not box0.any() or not box1.any():
    return None

  # find the intersection of 3D space
  ray0_o = cameras[0].rayo
  ray0_d = cameras[0].rayd

  ray0_o_topleft = ray0_o[0, :, box0[0], box0[1]]
  ray0_d_topleft = ray0_d[0, :, box0[0], box0[1]]

  ray0_o_bottomright = ray0_o[0, :, box0[2], box0[3]]
  ray0_d_bottomright = ray0_d[0, :, box0[2], box0[3]]

  ray0_o_bottomleft = ray0_o[0, :, box0[2], box0[1]]
  ray0_d_bottomleft = ray0_d[0, :, box0[2], box0[1]]

  ray0_o_topright = ray0_o[0, :, box0[0], box0[3]]
  ray0_d_topright = ray0_d[0, :, box0[0], box0[3]]

  ray1_o = cameras[1].rayo
  ray1_d = cameras[1].rayd

  ray1_o_topleft = ray1_o[0, :, box1[0], box1[1]]
  ray1_d_topleft = ray1_d[0, :, box1[0], box1[1]]

  ray1_o_bottomright = ray1_o[0, :, box1[2], box1[3]]
  ray1_d_bottomright = ray1_d[0, :, box1[2], box1[3]]

  ray1_o_bottomleft = ray1_o[0, :, box1[2], box1[1]]
  ray1_d_bottomleft = ray1_d[0, :, box1[2], box1[1]]

  ray1_o_topright = ray1_o[0, :, box1[0], box1[3]]
  ray1_d_topright = ray1_d[0, :, box1[0], box1[3]]

  topleft_intersect = _intersect_lines(
    ray0_o_topleft,
    ray0_d_topleft,
    ray1_o_topleft,
    ray1_d_topleft,
  )
  bottomright_intersect = _intersect_lines(
    ray0_o_bottomright,
    ray0_d_bottomright,
    ray1_o_bottomright,
    ray1_d_bottomright,
  )
  bottomleft_interset = _intersect_lines(
    ray0_o_bottomleft,
    ray0_d_bottomleft,
    ray1_o_bottomleft,
    ray1_d_bottomleft,
  )
  topright_intersect = _intersect_lines(
    ray0_o_topright,
    ray0_d_topright,
    ray1_o_topright,
    ray1_d_topright,
  )

  region3d = [topleft_intersect, bottomright_intersect, bottomleft_interset, topright_intersect]
  if None in region3d:
    # not a valid intersection, just drop this match
    return None

  region3d = torch.vstack(region3d)
  x_low = torch.min(region3d[:, 0]).item()
  y_low = torch.min(region3d[:, 1]).item()
  z_low = torch.min(region3d[:, 2]).item()

  x_high = torch.max(region3d[:, 0]).item()
  y_high = torch.max(region3d[:, 1]).item()
  z_high = torch.max(region3d[:, 2]).item()

  return (x_low, x_high), (y_low, y_high), (z_low, z_high)
