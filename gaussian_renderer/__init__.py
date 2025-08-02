#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import contextlib
import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(
  viewpoint_camera,
  pc: GaussianModel,
  pipe,
  bg_color: torch.Tensor,
  scaling_modifier=1.0,
  separate_sh=False,
  override_color=None,
  use_trained_exp=False,
  dropout_ratio: float = 0.0,
  dropout_by_opacity: bool = False,
  opacity_noise: float = 0.0,
):
  """Render the scene.

  Background tensor (bg_color) must be on GPU!
  """
  # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
  screenspace_points = (
    torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
  )
  with contextlib.suppress(Exception):
    screenspace_points.retain_grad()

  # Set up rasterization configuration
  tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
  tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

  raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,
    scale_modifier=scaling_modifier,
    viewmatrix=viewpoint_camera.world_view_transform,
    projmatrix=viewpoint_camera.full_proj_transform,
    sh_degree=pc.active_sh_degree,
    campos=viewpoint_camera.camera_center,
    prefiltered=False,
    debug=pipe.debug,
  )

  rasterizer = GaussianRasterizer(raster_settings=raster_settings)

  means3D = pc.get_xyz
  means2D = screenspace_points
  opacity = pc.get_opacity

  # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
  # scaling / rotation by the rasterizer.
  scales = None
  rotations = None
  cov3D_precomp = None

  if pipe.compute_cov3D_python:
    cov3D_precomp = pc.get_covariance(scaling_modifier)
  else:
    scales = pc.get_scaling
    rotations = pc.get_rotation

  # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
  # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
  shs = None
  colors_precomp = None
  if override_color is None:
    if pipe.convert_SHs_python:
      shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
      dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
      dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
      sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
      colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
      if separate_sh:
        dc, shs = pc.get_features_dc, pc.get_features_rest
      else:
        shs = pc.get_features
  else:
    colors_precomp = override_color

  # COAD: modify parameters here if required
  if dropout_ratio > 0.0:
    if dropout_by_opacity:
      opacity *= 1 - dropout_ratio
    else:
      dropout_mask = torch.rand(len(opacity), device=opacity.device) < (1 - dropout_ratio)
      means3D = means3D[dropout_mask]
      means2D = means2D[dropout_mask]
      shs = shs[dropout_mask]
      opacity = opacity[dropout_mask]
      scales = scales[dropout_mask]
      rotations = rotations[dropout_mask]
  if opacity_noise > 0.0:
    epsilon_opacity = torch.randn_like(opacity, device=opacity.device) * opacity_noise
    epsilon_opacity = torch.clamp(epsilon_opacity, min=-opacity_noise, max=opacity_noise)
    opacity = torch.clamp(opacity * (1.0 + epsilon_opacity), min=0.0, max=1.0)

  # Rasterize visible Gaussians to image, obtain their radii (on screen).
  _parameters = {
    "means3D": means3D,
    "means2D": means2D,
    "shs": shs,
    "colors_precomp": colors_precomp,
    "opacities": opacity,
    "scales": scales,
    "rotations": rotations,
    "cov3D_precomp": cov3D_precomp,
  }
  if separate_sh:
    _parameters["dc"] = dc

  rendered_image, radii, depth, alpha, contributors, contributions, top_contributor = rasterizer(
    **_parameters,
  )

  # Apply exposure to rendered image (training only)
  if use_trained_exp:
    exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
    rendered_image = (
      torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
      + exposure[:3, 3, None, None]
    )

  # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
  # They will be excluded from value updates used in the splitting criteria.
  rendered_image = rendered_image.clamp(0, 1)
  out = {
    "render": rendered_image,
    "viewspace_points": screenspace_points,
    "visibility_filter": (radii > 0).nonzero(),
    "radii": radii,
    "depth": depth,
    "rendered_depth": depth,
    "rendered_alpha": alpha,
    "contributors": contributors,
    "contributions": contributions,
    "top_contributor": top_contributor,
  }

  return out
