"""Visualizer for counting gaussians affecting certain pixel."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
  from scene.cameras import Camera
  from scene.gaussian_model import GaussianModel, MaskedGaussianModelView


def _calculate_2d_covariance(
  xyz: torch.Tensor,
  tanfov: tuple[float, float],
  focal: tuple[float, float],
  view_metric: torch.Tensor,
  covariance_3d: torch.Tensor,
) -> torch.Tensor:
  point_count = len(xyz)

  t = torch.matmul(
    view_metric.T[None, ...],
    torch.cat((xyz, torch.ones(size=(point_count, 1), device=xyz.device)), dim=-1)[..., None],
  )[:, :3, 0]

  lim = (1.3 * tanfov[0], 1.3 * tanfov[1])
  txtz = t[:, 0] / t[:, 2]
  tytz = t[:, 1] / t[:, 2]
  t[:, 0] = (
    txtz.maximum(torch.as_tensor(-lim[0], device=txtz.device)).minimum(
      torch.as_tensor(lim[0], device=txtz.device),
    )
    * t[:, 2]
  )
  t[:, 1] = (
    tytz.maximum(torch.as_tensor(-lim[1], device=tytz.device)).minimum(
      torch.as_tensor(lim[1], device=tytz.device),
    )
    * t[:, 2]
  )

  J0 = torch.stack(
    (
      focal[0] / t[:, 2],
      torch.zeros(point_count, device=t.device),
      -(focal[0] * t[:, 0]) / (t[:, 2] ** 2),
    ),
    dim=-1,
  )
  J1 = torch.stack(
    (
      torch.zeros(point_count, device=t.device),
      focal[1] / t[:, 2],
      -(focal[1] * t[:, 1]) / (t[:, 2] ** 2),
    ),
    dim=-1,
  )
  J2 = torch.zeros(size=(point_count, 3), device=J1.device)
  J = torch.stack((J0, J1, J2), dim=1)

  W = view_metric.T[None, :3, :3]

  T = torch.matmul(W, J)

  Vrk = torch.empty(size=(point_count, 3, 3), device=T.device)
  Vrk[:, 0, 0] = covariance_3d[:, 0]
  Vrk[:, 0, 1] = covariance_3d[:, 1]
  Vrk[:, 0, 2] = covariance_3d[:, 2]
  Vrk[:, 1, 0] = covariance_3d[:, 1]
  Vrk[:, 1, 1] = covariance_3d[:, 3]
  Vrk[:, 1, 2] = covariance_3d[:, 4]
  Vrk[:, 2, 0] = covariance_3d[:, 2]
  Vrk[:, 2, 1] = covariance_3d[:, 4]
  Vrk[:, 2, 2] = covariance_3d[:, 5]

  cov = torch.bmm(torch.bmm(T.permute(0, 2, 1), Vrk.permute(0, 2, 1)), T)
  result = torch.empty(size=(point_count, 3), device=cov.device)
  result[:, 0] = cov[:, 0, 0]
  result[:, 1] = cov[:, 0, 1]
  result[:, 2] = cov[:, 1, 1]
  return result


def _select_on_image(
  horizontals: torch.Tensor,
  verticals: torch.Tensor,
  radius: torch.Tensor,
  image_height: int,
  image_width: int,
) -> torch.Tensor:
  horizontal_selector = torch.logical_and(horizontals + radius >= 0, image_width > horizontals - radius)
  vertical_selector = torch.logical_and(verticals + radius >= 0, image_height > verticals - radius)
  return torch.logical_and(horizontal_selector, vertical_selector)


class _PreprocessResult(NamedTuple):
  image_size: tuple[int, int]  # width, height
  image_coordinate: torch.Tensor  # Nx2, x (horizontal, width related) and y (vertical, height related)
  depth: torch.Tensor  # N
  conic: torch.Tensor  # Nx3
  opacity: torch.Tensor  # N


def _preprocess(camera: Camera, gaussians: GaussianModel | MaskedGaussianModelView) -> _PreprocessResult:
  image_height, image_width = camera.image_height, camera.image_width
  total_gaussians = len(gaussians.get_xyz)
  device = gaussians.get_xyz.device
  tanfov = (math.tan(camera.FoVx * 0.5), math.tan(camera.FoVy * 0.5))
  focal = (image_width / (2 * tanfov[0]), image_height / (2 * tanfov[1]))

  # projection
  projected_xyz = torch.matmul(
    camera.full_proj_transform.T[None, ...],
    torch.cat((gaussians.get_xyz, torch.ones(size=(total_gaussians, 1), device=device)), dim=-1)[..., None],
  )[..., 0]
  projected_xyz = (projected_xyz / (projected_xyz[:, -1:] + 1e-10))[..., :-1]
  horizontals = ((projected_xyz[:, 0] + 1) * image_width - 1) / 2
  verticals = ((projected_xyz[:, 1] + 1) * image_height - 1) / 2
  horizontals = horizontals.round().long()
  verticals = verticals.round().long()

  # calculate depth
  depth = torch.mm(
    torch.cat((gaussians.get_xyz, torch.ones(size=(total_gaussians, 1), device=device)), dim=-1),
    camera.world_view_transform[:, 2:3],
  )[:, 0]

  covariance_3d = gaussians.get_covariance()

  covariance_2d = _calculate_2d_covariance(
    xyz=gaussians.get_xyz,
    tanfov=tanfov,
    focal=focal,
    view_metric=camera.world_view_transform,
    covariance_3d=covariance_3d,
  )

  _h_var = 0.3
  det_cov = covariance_2d[:, 0] * covariance_2d[:, 2] - covariance_2d[:, 1] ** 2
  covariance_2d[:, [0, 2]] += _h_var
  det_cov_plus_h_cov = covariance_2d[:, 0] * covariance_2d[:, 2] - covariance_2d[:, 1] ** 2
  h_convolution_scaling = torch.sqrt(
    (det_cov / det_cov_plus_h_cov).maximum(torch.as_tensor(0.000025, device=covariance_2d.device)),
  )
  det = det_cov_plus_h_cov

  selector = det != 0

  det_inv = 1 / det_cov_plus_h_cov
  conic = torch.stack(
    (
      covariance_2d[:, 2] * det_inv,
      -covariance_2d[:, 1] * det_inv,
      covariance_2d[:, 0] * det_inv,
    ),
    dim=-1,
  )
  opacity = gaussians.get_opacity.view(total_gaussians) * h_convolution_scaling

  # select points that does hit the image plane
  mid = (covariance_2d[:, 0] + covariance_2d[:, 2]) / 2
  lambda1 = mid + torch.sqrt((mid**2 - det).maximum(torch.as_tensor(0.1, device=det.device)))
  lambda2 = mid - torch.sqrt((mid**2 - det).maximum(torch.as_tensor(0.1, device=det.device)))
  radius = torch.ceil(torch.sqrt(torch.max(lambda1, lambda2)) * 3)
  selector = torch.logical_and(
    selector,
    _select_on_image(
      horizontals=horizontals,
      verticals=verticals,
      radius=radius,
      image_height=image_height,
      image_width=image_width,
    ),
  )

  return _PreprocessResult(
    image_size=(image_width, image_height),
    image_coordinate=torch.stack((horizontals, verticals), dim=-1)[selector],
    depth=depth[selector],
    conic=conic[selector],
    opacity=opacity[selector],
  )


def _count(
  data: _PreprocessResult,
  *,
  effective: bool = False,
) -> torch.Tensor:
  image_width, image_height = data.image_size
  # sort points globally
  depth, indices = torch.sort(data.depth, descending=True)
  image_coordinate = data.image_coordinate[indices]
  conic = data.conic[indices]
  opacity = data.opacity[indices]

  result = torch.zeros(size=(image_width, image_height), dtype=torch.long, device=depth.device)
  coordinate_helper = torch.stack(
    (
      torch.arange(image_width, device=conic.device)[:, None].expand(-1, image_height),
      torch.arange(image_height, device=conic.device)[None, :].expand(image_width, -1),
    ),
    dim=-1,
  )  # width * height * 2

  if not effective:
    _batch_size = 100
    for i in range(0, len(depth), _batch_size):
      sl = slice(i, i + _batch_size)
      distances = image_coordinate[sl, None, None, :] - coordinate_helper[None, ...]
      powers = (
        -0.5
        * (
          conic[sl, 0, None, None] * (distances[..., 0] ** 2)
          + conic[sl, 2, None, None] * (distances[..., 1] ** 2)
        )
        - conic[sl, 1, None, None] * distances[..., 0] * distances[..., 1]
      )
      effective_selector = powers <= 0
      alpha = (opacity[sl, None, None] * torch.exp(powers)).minimum(
        torch.as_tensor(0.99, device=powers.device),
      )
      effective_selector = torch.logical_and(effective_selector, alpha >= 1 / 255)
      result += effective_selector.sum(dim=0)

  return result

def count(
  gaussian: GaussianModel | MaskedGaussianModelView,
  camera: Camera,
) -> torch.Tensor:
  preprocess_result = _preprocess(camera=camera, gaussians=gaussian)
  counting = _count(data=preprocess_result)
  return counting.T

