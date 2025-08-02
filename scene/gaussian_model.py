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
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from scene.densification_hook_typing import (
  HookManager,
  NewGaussianPack,
  PendingDensify,
  PendingDensifyDetails,
)
from scene.densification_hooks import install_hooks
from scene.extra_metrics_manager import ExtraMetricsManager, MetricProjectionSpecification
from utils.general_utils import (
  build_rotation,
  build_scaling_rotation,
  get_expon_lr_func,
  inverse_sigmoid,
  strip_symmetric,
)
from utils.registers import registers
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

if TYPE_CHECKING:
  from collections.abc import Callable

  from arguments import OptimizationParameters
  from scene.densification_classifiers_typing import GaussianADCClassifier
  from utils.graphics_utils import BasicPointCloud

  from .cameras import Camera

with contextlib.suppress(Exception):
  from diff_gaussian_rasterization import SparseGaussianAdam


class GaussianModel:
  def setup_functions(self):
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
      L = build_scaling_rotation(scaling_modifier * scaling, rotation)
      actual_covariance = L @ L.transpose(1, 2)
      symm = strip_symmetric(actual_covariance)
      return symm

    self.scaling_activation = torch.exp
    self.scaling_inverse_activation = torch.log

    self.covariance_activation = build_covariance_from_scaling_rotation

    self.opacity_activation = torch.sigmoid
    self.inverse_opacity_activation = inverse_sigmoid

    self.rotation_activation = torch.nn.functional.normalize

  def capture_trace(self) -> tuple[torch.Tensor, ...]:
    grad = self.xyz_gradient_accum / self.denom
    grad[grad.isnan()] = 0.0
    extra_metrics = self.extra_metrics.capture()
    raw_position = self.get_xyz
    position_vectors = raw_position - registers.point_center
    position_vector_lengths = torch.linalg.norm(position_vectors, dim=-1, keepdims=True)
    normalized_position = (
      position_vectors
      / position_vector_lengths
      * (position_vector_lengths / registers.point_radius).clamp(min=0, max=1)
    )
    return normalized_position, self.get_scaling, self.get_rotation, self.get_opacity, grad, extra_metrics

  def __init__(self, sh_degree, optimizer_type="default", *, train_mode: bool = False):
    self.active_sh_degree = 0
    self.optimizer_type = optimizer_type
    self.max_sh_degree = sh_degree
    self._xyz = torch.empty(0)
    self._features_dc = torch.empty(0)
    self._features_rest = torch.empty(0)
    self._scaling = torch.empty(0)
    self._rotation = torch.empty(0)
    self._opacity = torch.empty(0)
    self.max_radii2D = torch.empty(0)
    self.xyz_gradient_accum = torch.empty(0)
    self.denom = torch.empty(0)
    self.optimizer = None
    self.percent_dense = 0
    self.spatial_lr_scale = 0
    self.setup_functions()
    self.registers = {}

    if train_mode:
      self.densify_hook_manager = HookManager()
      self.extra_metrics = ExtraMetricsManager()
      self._densify_classifiers: dict[str, GaussianADCClassifier] = {}
      self._prune_classifiers: dict[str, GaussianADCClassifier] = {}

      # This member is used in densify classifiers
      #  If visualization is requested, this member will be initialized and managed by the ADCVisualizerHook
      self.adc_visualizer = None

      self._current_id = 0
      self.point_id = torch.empty(0)

      def _issue_ids(*, count: int) -> torch.Tensor:
        result = torch.arange(count).long() + self._current_id
        self._current_id += count
        return result.cuda()

      object.__setattr__(self, "_issue_ids", _issue_ids)

  def finish_configuration(self) -> None:
    install_hooks(target=self)

  def capture(self) -> tuple:
    return (
      self.active_sh_degree,
      self._xyz,
      self._features_dc,
      self._features_rest,
      self._scaling,
      self._rotation,
      self._opacity,
      self.max_radii2D,
      self.xyz_gradient_accum,
      self.denom,
      self.spatial_lr_scale,
      self.registers,
      self._current_id,
      self.point_id,
      self._exposure,
      self.extra_metrics.capture_state(),
      self.optimizer.state_dict(),
    )

  def restore(self, model_args: tuple, training_args: OptimizationParameters) -> None:
    # save xyz_gradient_accum and denom into temporary variable since the value will be overwritten during
    #  training_setup if stored directly into self.xyz_gradient_accum and self.denom
    #  see also comments below
    (
      self.active_sh_degree,
      self._xyz,
      self._features_dc,
      self._features_rest,
      self._scaling,
      self._rotation,
      self._opacity,
      self.max_radii2D,
      xyz_gradient_accum,
      denom,
      self.spatial_lr_scale,
      self.registers,
      self._current_id,
      self.point_id,
      self._exposure,
      extra_metrics_state,
      opt_dict,
    ) = model_args
    # place training_setup before restoring state of extra_metrics
    #  training_setup uses extra_metrics.reset internally, which nullifies the effect of restore
    self.training_setup(training_args)
    self.xyz_gradient_accum = xyz_gradient_accum
    self.denom = denom
    self.extra_metrics.restore_state(state=extra_metrics_state)
    self.optimizer.load_state_dict(opt_dict)

  @property
  def get_scaling(self):
    return self.scaling_activation(self._scaling)

  @property
  def get_rotation(self):
    return self.rotation_activation(self._rotation)

  @property
  def get_xyz(self):
    return self._xyz

  @property
  def get_features(self):
    features_dc = self._features_dc
    features_rest = self._features_rest
    return torch.cat((features_dc, features_rest), dim=1)

  @property
  def get_features_dc(self):
    return self._features_dc

  @property
  def get_features_rest(self):
    return self._features_rest

  @property
  def get_opacity(self):
    return self.opacity_activation(self._opacity)

  @property
  def get_exposure(self):
    return self._exposure

  def get_exposure_from_name(self, image_name):
    if self.pretrained_exposures is None:
      return self._exposure[self.exposure_mapping[image_name]]
    else:
      return self.pretrained_exposures[image_name]

  def get_covariance(self, scaling_modifier=1):
    return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

  def oneupSHdegree(self):
    if self.active_sh_degree < self.max_sh_degree:
      self.active_sh_degree += 1

  def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float):
    self.spatial_lr_scale = spatial_lr_scale
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialization : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = self.inverse_opacity_activation(
      0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
    )

    self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    self.point_id = self._issue_ids(count=len(self._xyz))
    self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    self._scaling = nn.Parameter(scales.requires_grad_(True))
    self._rotation = nn.Parameter(rots.requires_grad_(True))
    self._opacity = nn.Parameter(opacities.requires_grad_(True))
    self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
    self.pretrained_exposures = None
    exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
    self._exposure = nn.Parameter(exposure.requires_grad_(True))

  def training_setup(self, training_args):
    self.percent_dense = training_args.percent_dense
    self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    self.extra_metrics.reset(points=len(self.get_xyz))

    l = [
      {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
      {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
      {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
      {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
      {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
      {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
    ]

    if self.optimizer_type == "default":
      self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    elif self.optimizer_type == "sparse_adam":
      try:
        self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
      except:
        # A special version of the rasterizer is required to enable sparse adam
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    self.exposure_optimizer = torch.optim.Adam([self._exposure])

    print(len(self.get_xyz), len(self.xyz_gradient_accum), len(self._features_dc), len(self._features_rest))

    self.xyz_scheduler_args = get_expon_lr_func(
      lr_init=training_args.position_lr_init * self.spatial_lr_scale,
      lr_final=training_args.position_lr_final * self.spatial_lr_scale,
      lr_delay_mult=training_args.position_lr_delay_mult,
      max_steps=training_args.position_lr_max_steps,
    )

    self.exposure_scheduler_args = get_expon_lr_func(
      training_args.exposure_lr_init,
      training_args.exposure_lr_final,
      lr_delay_steps=training_args.exposure_lr_delay_steps,
      lr_delay_mult=training_args.exposure_lr_delay_mult,
      max_steps=training_args.iterations,
    )

  def update_learning_rate(self, iteration):
    """Learning rate scheduling per step"""
    if self.pretrained_exposures is None:
      for param_group in self.exposure_optimizer.param_groups:
        param_group["lr"] = self.exposure_scheduler_args(iteration)

    for param_group in self.optimizer.param_groups:
      if param_group["name"] == "xyz":
        lr = self.xyz_scheduler_args(iteration)
        param_group["lr"] = lr
        return lr

  def construct_list_of_attributes(self):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
      l.append("f_dc_{}".format(i))
    for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
      l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(self._scaling.shape[1]):
      l.append("scale_{}".format(i))
    for i in range(self._rotation.shape[1]):
      l.append("rot_{}".format(i))
    return l

  def save_ply(
    self,
    path: str,
    mappers: dict[str, Callable[[np.typing.NDArray[np.float32]], np.float32]] | None,
  ) -> None:
    """Save the model to given path.

    Mapping functions can be passed to transform parameters before they are saved, which will not affect
     the model in memory.
    """
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    xyz = self._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self._opacity.detach().cpu().numpy()
    scale = self._scaling.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()

    if mappers is not None:
      if "xyz" in mappers:
        xyz = mappers["xyz"](xyz)
      if "f_dc" in mappers:
        f_dc = mappers["f_dc"](f_dc)
      if "f_rest" in mappers:
        f_rest = mappers["f_rest"](f_rest)
      if "opacities" in mappers:
        opacities = mappers["opacities"](opacities)
      if "scale" in mappers:
        scale = mappers["scale"](scale)
      if "rotation" in mappers:
        rotation = mappers["rotation"](rotation)

    dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)

  def reset_opacity(self):
    opacities_new = self.inverse_opacity_activation(
      torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
    )
    optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    self._opacity = optimizable_tensors["opacity"]

  def load_ply(self, path, use_train_test_exp=False):
    plydata = PlyData.read(path)
    if use_train_test_exp:
      exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
      if os.path.exists(exposure_file):
        with open(exposure_file, "r") as f:
          exposures = json.load(f)
        self.pretrained_exposures = {
          image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda()
          for image_name in exposures
        }
        print(f"Pretrained exposures loaded.")
      else:
        print(f"No exposure to be loaded at {exposure_file}")
        self.pretrained_exposures = None

    xyz = np.stack(
      (
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"]),
      ),
      axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
      features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
      scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
      rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    self._features_dc = nn.Parameter(
      torch.tensor(features_dc, dtype=torch.float, device="cuda")
      .transpose(1, 2)
      .contiguous()
      .requires_grad_(True)
    )
    self._features_rest = nn.Parameter(
      torch.tensor(features_extra, dtype=torch.float, device="cuda")
      .transpose(1, 2)
      .contiguous()
      .requires_grad_(True)
    )
    self._opacity = nn.Parameter(
      torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
    )
    self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    self.active_sh_degree = self.max_sh_degree

  def replace_tensor_to_optimizer(self, tensor, name):
    optimizable_tensors = {}
    for group in self.optimizer.param_groups:
      if group["name"] == name:
        stored_state = self.optimizer.state.get(group["params"][0], None)
        stored_state["exp_avg"] = torch.zeros_like(tensor)
        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

        del self.optimizer.state[group["params"][0]]
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        self.optimizer.state[group["params"][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors

  def _prune_optimizer(self, mask):
    optimizable_tensors = {}
    for group in self.optimizer.param_groups:
      stored_state = self.optimizer.state.get(group["params"][0], None)
      if stored_state is not None:
        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

        del self.optimizer.state[group["params"][0]]
        group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
        self.optimizer.state[group["params"][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors

  def prune_points(self, selector: torch.Tensor) -> None:
    """Prune gaussians being selected by the selector."""
    valid_points_mask = ~selector
    optimizable_tensors = self._prune_optimizer(valid_points_mask)

    self._xyz = optimizable_tensors["xyz"]
    self._features_dc = optimizable_tensors["f_dc"]
    self._features_rest = optimizable_tensors["f_rest"]
    self._opacity = optimizable_tensors["opacity"]
    self._scaling = optimizable_tensors["scaling"]
    self._rotation = optimizable_tensors["rotation"]

    self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
    self.denom = self.denom[valid_points_mask]
    self.extra_metrics.filter_out(mask=selector)

    self.max_radii2D = self.max_radii2D[valid_points_mask]
    self.point_id = self.point_id[valid_points_mask]

  def cat_tensors_to_optimizer(self, tensors_dict):
    optimizable_tensors = {}
    for group in self.optimizer.param_groups:
      assert len(group["params"]) == 1
      extension_tensor = tensors_dict[group["name"]]
      stored_state = self.optimizer.state.get(group["params"][0], None)
      if stored_state is not None:
        stored_state["exp_avg"] = torch.cat(
          (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
        )
        stored_state["exp_avg_sq"] = torch.cat(
          (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
        )

        del self.optimizer.state[group["params"][0]]
        group["params"][0] = nn.Parameter(
          torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
        )
        self.optimizer.state[group["params"][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(
          torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
        )
        optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors

  def _append_gaussians(self, new_gaussians: NewGaussianPack) -> None:
    d = {
      "xyz": new_gaussians.xyz,
      "f_dc": new_gaussians.features_dc,
      "f_rest": new_gaussians.features_rest,
      "opacity": new_gaussians.opacity,
      "scaling": new_gaussians.scaling,
      "rotation": new_gaussians.rotation,
    }

    optimizable_tensors = self.cat_tensors_to_optimizer(d)
    self._xyz = optimizable_tensors["xyz"]
    self._features_dc = optimizable_tensors["f_dc"]
    self._features_rest = optimizable_tensors["f_rest"]
    self._opacity = optimizable_tensors["opacity"]
    self._scaling = optimizable_tensors["scaling"]
    self._rotation = optimizable_tensors["rotation"]

    self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    self.extra_metrics.reset(points=len(self.get_xyz))

    self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    self.point_id = torch.cat((self.point_id, new_gaussians.point_id))

  def _densify_by_split(self, selector: torch.Tensor, n: int = 2) -> NewGaussianPack:
    stds = self.get_scaling[selector].repeat(n, 1)
    means = torch.zeros((stds.size(0), 3), device=stds.device)
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(self._rotation[selector]).repeat(n, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selector].repeat(n, 1)
    new_ids = self._issue_ids(count=len(new_xyz))

    self.registers["split-mapping"] = torch.cat(
      (self.point_id[selector][:, None], new_ids.view(-1, n)),
      dim=-1,
    )

    return NewGaussianPack(
      point_id=new_ids,
      xyz=new_xyz,
      features_dc=self._features_dc[selector].repeat(n, 1, 1),
      features_rest=self._features_rest[selector].repeat(n, 1, 1),
      opacity=self._opacity[selector].repeat(n, 1),
      scaling=self.scaling_inverse_activation(self.get_scaling[selector].repeat(n, 1) / (0.8 * n)),
      rotation=self._rotation[selector].repeat(n, 1),
    )

  def _densify_by_clone(self, selector: torch.Tensor) -> NewGaussianPack:
    new_ids = self._issue_ids(count=selector.sum().item())

    self.registers["clone-mapping"] = torch.cat(
      (
        self.point_id[selector][:, None],
        self.point_id[selector][:, None],
        new_ids[:, None],
      ),
      dim=-1,
    )

    return NewGaussianPack(
      point_id=new_ids,
      xyz=self._xyz[selector],
      features_dc=self._features_dc[selector],
      features_rest=self._features_rest[selector],
      opacity=self._opacity[selector],
      scaling=self._scaling[selector],
      rotation=self._rotation[selector],
    )

  # rewrite densify procedure
  #  original implementation calculates the selector for multiple times, modifies the list of gaussians in
  #   the middle of densify and prune procedure, which is a practice that should be avoided
  #  the rewritten implementation will:
  #   calculate all criteria used in densification in method densify_and_prune, generating masks (selectors)
  #    that selects all gaussians to be cloned, split and pruned
  #   densify_and_clone and densify_and_split are now renamed into densify_by_clone and densify_by_split,
  #    which simply take the selector and clone/split gaussians selected by it

  @torch.no_grad()
  def densify_and_prune(self, extent: torch.Tensor) -> None:
    self.densify_hook_manager.before_densify_selection(gaussian=self)

    # apply densify classifiers
    densify_selector = self._apply_classifiers(
      classifiers=self._densify_classifiers,
      phase="densify",
    )

    # decide gaussians to clone/split
    _smaller_selector = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent
    clone_selector = densify_selector.logical_and(_smaller_selector)
    split_selector = densify_selector.logical_and(~_smaller_selector)
    # collect densified gaussians
    clone_result = self._densify_by_clone(selector=clone_selector)
    split_result = self._densify_by_split(selector=split_selector)
    new_gaussians = len(clone_result.xyz) + len(split_result.xyz)

    self.densify_hook_manager.before_densify_applied(
      gaussian=self,
      points=PendingDensify(
        clone=PendingDensifyDetails(selector=clone_selector, new_gaussian=clone_result),
        split=PendingDensifyDetails(selector=split_selector, new_gaussian=split_result),
      ),
    )

    # apply new gaussians
    self._append_gaussians(new_gaussians=clone_result)
    self._append_gaussians(new_gaussians=split_result)

    self.densify_hook_manager.before_prune_selection(gaussian=self)

    # apply prune classifiers
    prune_selector = self._apply_classifiers(
      classifiers=self._prune_classifiers,
      phase="prune",
    )

    self.densify_hook_manager.before_prune_applied(
      gaussian=self,
      selector=prune_selector,
    )

    # remove pruned gaussians
    self.prune_points(
      selector=prune_selector.logical_or(
        torch.cat(
          (split_selector, torch.zeros(size=(new_gaussians,), device=split_selector.device, dtype=bool)),
        ),
      ),
    )
    torch.cuda.empty_cache()

    self.densify_hook_manager.before_done(gaussian=self)

  def _apply_classifiers(
    self,
    classifiers: dict[str, GaussianADCClassifier],
    phase: Literal["densify", "prune"],
  ) -> torch.Tensor:
    selector = None
    for name, classifier in classifiers.items():
      _selector = classifier(gaussians=self, visualizer=self.adc_visualizer)
      (
        self.densify_hook_manager.after_each_densify_classifier
        if phase == "densify"
        else self.densify_hook_manager.after_each_prune_classifier
      )(
        gaussian=self,
        name=name,
        selector=_selector,
      )
      selector = _selector if selector is None else selector.logical_or(_selector)
    return selector

  def add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradient_accum[update_filter] += torch.norm(
      viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
    )
    self.denom[update_filter] += 1

  def add_extra_densification_stats_old(
    self,
    bboxes: tuple[torch.Tensor, torch.Tensor],
    segments: tuple[torch.Tensor, torch.Tensor],
    cameras: tuple[Camera, Camera],
    metrics: tuple[MetricProjectionSpecification, ...],
    *,
    lhs_only: bool,
  ) -> None:
    self.extra_metrics.project_old(
      position=self.get_xyz,
      boxes=bboxes,
      segments=segments,
      cameras=cameras,
      metrics=metrics,
      lhs_only=lhs_only,
    )

  def add_extra_densification_stats(
    self,
    camera: Camera,
    metrics: tuple[MetricProjectionSpecification, ...],
  ) -> None:
    self.extra_metrics.project(
      gaussian=self,
      camera=camera,
      metrics=metrics,
    )

  def register_densify_classifier(self, name: str, classifier: GaussianADCClassifier) -> None:
    self._densify_classifiers[name] = classifier

  def register_prune_classifier(self, name: str, classifier: GaussianADCClassifier) -> None:
    self._prune_classifiers[name] = classifier

  class ParameterPack(TypedDict):
    """Named parameter pack."""

    xyz: nn.Parameter
    features_dc: nn.Parameter
    features_rest: nn.Parameter
    scaling: nn.Parameter
    rotation: nn.Parameter
    opacity: nn.Parameter
    exposure: nn.Parameter

  def parameters(self) -> GaussianModel.ParameterPack:
    return GaussianModel.ParameterPack(
      xyz=self._xyz,
      features_dc=self._features_dc,
      features_rest=self._features_rest,
      scaling=self._scaling,
      rotation=self._rotation,
      opacity=self._opacity,
      exposure=self._exposure,
    )
