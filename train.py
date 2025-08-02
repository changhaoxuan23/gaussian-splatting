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

import itertools
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from functools import partial
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from random import choice, randint, sample
from sys import _getframe
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from moge.model.v2 import MoGeModel
from torchvision.utils import save_image

from arguments import ModelParameters, OptimizationParameters, PipelineParameters, TrainParameters
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.densification_classifiers import attach_classifier
from scene.densify_classifiers.window_based_densify import WindowSources
from scene.extra_metrics_manager import MetricProjectionSpecification
from utils.camera_utils import set_rays_od
from utils.general_debugging import export_traceback
from utils.general_utils import get_expon_lr_func, safe_state
from utils.graphics_utils import BasicPointCloud
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.modifications import modifications
from utils.modifications import prepare_parser as prepare_modifications
from utils.registers import registers
from utils.resnet_encoder import encode_without_resize
from utils.statistics import full_statistics
from utils.timer import Timer
from utils.train_progress_manager import TrainingProgressManager
from utils.video_encoder import VideoEncoder

sys.path.append(str(Path.cwd().parent / "vggt_depth"))
from vggt_depth.mv_depth_infer import vggt_inference

if TYPE_CHECKING:
  from collections.abc import Callable, Iterable, Sequence

  from scene.cameras import Camera

try:
  from torch.utils.tensorboard import SummaryWriter

  TENSORBOARD_FOUND = True
except ImportError:
  TENSORBOARD_FOUND = False

try:
  from fused_ssim import fused_ssim

  FUSED_SSIM_AVAILABLE = True
except ImportError:
  FUSED_SSIM_AVAILABLE = False

SPARSE_ADAM_AVAILABLE = find_spec("diff_gaussian_rasterization.SparseGaussianAdam") is not None


def _setup_classifiers(
  gaussian: GaussianModel,
  optimization_parameters: OptimizationParameters,
  training_cameras: Sequence[Camera],
  output_directory: Path,
  scene: Scene,
) -> dict[str, Any]:
  new_locals: dict[str, Any] = {}
  if not modifications.get("no_grad_based", False):
    attach_classifier(
      target=gaussian,
      mode="densify",
      classifier="grad_based_densify",
      optimization_config=optimization_parameters,
    )

  if any(modifications.get(mod, False) for mod in ["projection", "new-projection"]):
    attach_classifier(
      target=gaussian,
      mode="densify",
      classifier="ldl_based_densify",
      optimization_config=optimization_parameters,
    )

  if any(modifications.get(mod, False) for mod in ["mvgs-like-w2w", "mvgs-like-w2b"]):
    window_sources = WindowSources(box_pairs=[], cameras=[])
    attach_classifier(
      target=gaussian,
      mode="densify",
      classifier="window_based_densify",
      source=window_sources,
    )
    new_locals["window_sources"] = window_sources

  if modifications.get("random-clone-split", False):
    attach_classifier(
      target=gaussian,
      mode="densify",
      classifier="random_densify",
      ratio=0.1,
    )

  if "trace" in modifications and modifications["trace"].get("capture", False):
    attach_classifier(
      target=gaussian,
      mode="densify",
      classifier="random_densify",
      ratio=0.025,
    )

  if "trace" in modifications and "model" in modifications["trace"]:
    attach_classifier(
      target=gaussian,
      mode="densify",
      classifier="predictor_based_densify",
      thresholds=(0, 0, 0),
      training_progress=registers.train_progress,
      output_directory=registers.out_path,
    )

  if "foreseer" in modifications:
    attach_classifier(
      target=gaussian,
      mode="densify",
      classifier="foreseer_densify",
      optimization_config=optimization_parameters,
      training_progress=registers.train_progress,
      use_gradient=modifications["foreseer"].get("explore-with-gaussian-gradient", True),
      random_ratio=modifications["foreseer"].get("explore-with-random", 0),
      metrics_to_use=[
        item
        for item in [
          name if modifications["foreseer"].get(f"use-{name}", True) else None
          for name in ("dldl", "dl1", "dssim")
        ]
        if item is not None
      ],
      strict=modifications["foreseer"].get("strict", False),
      target_gaussian=modifications["foreseer"].get("target-gaussian", 0),
      visualize_views=modifications["foreseer"].get("visualize-views", 0),
      cameras=training_cameras,
      output_directory=output_directory,
      temperature_modifier=modifications["foreseer"].get("temperature-modifier", 0.0),
      clustering_distance=modifications["foreseer"].get("clustering", 0.0),
      neighbor_radius=scene.cameras_extent * modifications["foreseer"].get("neighbor-ratio", 0.0),
    )

  attach_classifier(
    target=gaussian,
    mode="prune",
    classifier="opacity_based_prune",
    threshold=0.005,
  )

  return new_locals


def training(
  dataset: ModelParameters,
  opt: OptimizationParameters,
  pipe: PipelineParameters,
  training_parameters: TrainParameters,
):
  if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
    sys.exit(
      "Trying to use sparse adam but it is not installed. "
      "Please install the correct rasterizer using pip install [3dgs_accel].",
    )

  sam_rho = 0 if "sam" not in modifications else modifications.get("rho", 0)
  sam_opacity_only = False if "sam" not in modifications else modifications.get("opacity-only", False)

  # COAD hyperparameters
  if "coad" in modifications:
    coad_dropout = modifications["coad"].get("dropout", 0.0)
    coad_opacity_noise = modifications["coad"].get("opacity_noise", 0.0)
  else:
    coad_dropout = 0.0
    coad_opacity_noise = 0.0

  registers.register("render", render)

  if any(mod in modifications for mod in ["projection", "multiplier"]):
    ema_loss_difference = {}
    ema_loss_history = {}
  if "ldl" in modifications.get("metrics-to-trace", ()):
    current_loss = {}
    last_loss = {}
  if "random-clone-split" in modifications:
    opt.densification_interval = 400

  first_iter = 0
  tb_writer = prepare_output_and_logger(dataset)

  if "save-metrics" in modifications:
    registers.out_path.joinpath("saved-metrics").mkdir(exist_ok=True)

  gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, train_mode=True)
  scene = Scene(dataset, gaussians)
  registers.point_center = torch.as_tensor(registers.point_center, device="cuda")

  gaussians.training_setup(opt)
  if training_parameters.start_checkpoint is not None:
    (model_params, first_iter) = torch.load(training_parameters.start_checkpoint)
    gaussians.restore(model_params, opt)

  bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
  background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

  iter_start = torch.cuda.Event(enable_timing=True)
  iter_end = torch.cuda.Event(enable_timing=True)

  use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
  depth_l1_weight = get_expon_lr_func(
    opt.depth_l1_weight_init,
    opt.depth_l1_weight_final,
    max_steps=opt.iterations,
  )

  ema_loss_for_log = 0.0
  ema_l_l1depth_for_log = 0.0

  training_progress = TrainingProgressManager(
    start_iteration=first_iter,
    optimization=opt,
    training=training_parameters,
  )
  registers.register("train_progress", training_progress)
  progress_bar = training_progress.progress_bar

  set_rays_od(scene.getTrainCameras())
  registers.register("training_cameras", scene.getTrainCameras())

  new_locals = _setup_classifiers(
    gaussian=gaussians,
    optimization_parameters=opt,
    training_cameras=scene.getTrainCameras(),
    output_directory=dataset.model_path,
    scene=scene,
  )
  _getframe().f_locals.update(new_locals)
  # make static checkers happy
  if False:
    window_sources = None

  gaussians.finish_configuration()

  if "external-depth" in modifications:
    _cameras = scene.getTrainCameras()
    _image_height = _cameras[0].image_height
    _image_width = _cameras[0].image_width

    _external_depth = modifications["external-depth"]
    _depth_source = _external_depth.get("source", "")

    if _depth_source == "moge":
      moge_model = MoGeModel.from_pretrained(Path(__file__).parent / "model.pt").cuda()
      input_images = torch.stack(tuple(camera.original_image.cuda() for camera in _cameras), dim=0)
      camera_fovx = torch.as_tensor([camera.FoVx for camera in _cameras], device="cuda")
      external_depths = moge_model.infer(image=input_images, fov_x=camera_fovx)["depth"].clone()
      del moge_model
    elif _depth_source == "vggt":
      _temporary_images: list[BytesIO] = []
      for camera in _cameras:
        _buffer = BytesIO()
        save_image(camera.original_image, _buffer, format="PNG")
        _buffer.seek(0)
        _temporary_images.append(_buffer)
      use_vggt_point_cloud = _external_depth.get("use-vggt-point-cloud", False)
      external_depths, _, _, _, vggt_point_cloud = vggt_inference(
        images=_temporary_images,
        out_size=(_image_height, _image_width),
        include_point_cloud=use_vggt_point_cloud,
      )
      if use_vggt_point_cloud and vggt_point_cloud is not None:
        gaussians.create_from_pcd(
          pcd=BasicPointCloud(points=vggt_point_cloud.points, colors=vggt_point_cloud.colors, normals=None),
          cam_infos=scene.getTrainCameras(),
          spatial_lr_scale=gaussians.spatial_lr_scale,
        )
        gaussians.training_setup(opt)
      del _temporary_images
    else:
      message = f"unknown external depth source: {_depth_source}"
      raise ValueError(message)

    for camera, depth in zip(_cameras, external_depths, strict=True):
      camera.external_depth = depth
      from pdb import set_trace
      set_trace()
      _statistics_loss = _external_depth.get("statistics-loss")
      if _statistics_loss is not None and _statistics_loss.get("samples") is None:
        camera.external_statistics = full_statistics(
          depth[None, ...],
          patch_size=_statistics_loss.get("patch-size", 28),
        )

  sample_camera = scene.getTrainCameras()[0]
  training_monitor = VideoEncoder(
    width=sample_camera.image_width * len(scene.getTrainCameras()),
    height=sample_camera.image_height,
    destination=registers.out_path / "train_monitor.mkv",
    logging=registers.out_path / "ffmpeg.log",
  )

  core_training_timer = Timer()
  while training_progress.step():
    if training_progress.should_start_debug:
      pipe.debug = True

    iter_start.record()

    gaussians.update_learning_rate(training_progress.iteration)

    # Every 1000 its we increase the levels of SH up to a maximum degree
    if training_progress.should_increase_sh_level:
      gaussians.oneupSHdegree()

    if "mvgs-like-w2w" in modifications or "mvgs-like-w2b" in modifications:
      loss_storage = []

    # Pick random Camera(s)
    if "with-camera-grouping" in modifications:
      if training_progress.should_regroup_camera:
        scene.rebuild_train_camera_groups()
      group = choice(scene.getTrainCameraGroups())  # noqa: S311 -- not for cryptographic usage
    else:
      group = scene.getTrainCameras()
    cams = sample(group, k=min(pipe.mv, len(group)))

    class _ForwardPassResult(NamedTuple):
      loss: torch.Tensor  # calculated loss
      images: list[torch.Tensor]  # list of rendered images
      viewspace_point_tensors: list[torch.Tensor]
      visibility_filters: list[torch.Tensor]
      radiis: list[torch.Tensor]
      mean_depth_loss: torch.Tensor
      mean_l1_loss: torch.Tensor

    def _forward_pass(cams: Iterable[Camera]) -> _ForwardPassResult:
      total_loss = 0
      images = []
      viewspace_point_tensors = []
      visibility_filters = []
      radiis = []
      l1_losses = []
      depth_losses = []
      for cam in cams:
        # render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(
          cam,
          gaussians,
          pipe,
          bg,
          use_trained_exp=dataset.train_test_exp,
          separate_sh=SPARSE_ADAM_AVAILABLE,
          dropout_ratio=coad_dropout,
          dropout_by_opacity=False,
          opacity_noise=coad_opacity_noise,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
          render_pkg["render"],
          render_pkg["viewspace_points"],
          render_pkg["visibility_filter"],
          render_pkg["radii"],
        )
        images.append(image)
        viewspace_point_tensors.append(viewspace_point_tensor)
        visibility_filters.append(visibility_filter)
        radiis.append(radii)

        if cam.alpha_mask is not None:
          alpha_mask = cam.alpha_mask.cuda()
          image *= alpha_mask

        # calculate loss
        gt_image = cam.original_image.cuda()
        l_l1 = l1_loss(image, gt_image)
        l1_losses.append(l_l1)
        if FUSED_SSIM_AVAILABLE:
          ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
          ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * l_l1 + opt.lambda_dssim * (1.0 - ssim_value)
        # Depth regularization
        l_l1depth_pure = 0.0
        if depth_l1_weight(training_progress.iteration) > 0 and cam.depth_reliable:
          inv_depth = render_pkg["depth"]
          mono_invdepth = cam.invdepthmap.cuda()
          depth_mask = cam.depth_mask.cuda()

          l_l1depth_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
          l_l1depth = depth_l1_weight(training_progress.iteration) * l_l1depth_pure
          loss += l_l1depth
          l_l1depth = l_l1depth.item()
        else:
          l_l1depth = 0
        depth_losses.append(l_l1depth)

        if "external-depth" in modifications and training_progress.should_apply_external_depth:
          _external_depth = modifications.get("external-depth")
          original_depth = depth

          # alignment
          _depth_alignment = _external_depth.get("align", "none")
          if _depth_alignment == "mean":
            _ratio = depth.mean() / camera.external_depth.mean()
            external_depth = camera.external_depth * _ratio
          elif _depth_alignment == "mean-variance":
            mu_source = camera.external_depth.mean()
            mu_target = depth.mean()
            var_source = (camera.external_depth**2).mean() - mu_source**2
            var_target = (depth**2).mean() - mu_target**2
            _ratio = torch.sqrt(var_target / var_source)
            _bias = mu_target - _ratio * mu_source
            external_depth = camera.external_depth * _ratio + _bias
          elif _depth_alignment == "none":
            external_depth = camera.external_depth
          else:
            message = f"Invalid align method: {_depth_alignment}"
            raise ValueError(message)
          external_depth = external_depth.detach()

          original_depth = original_depth[None, ...]

          if "plain-loss" in _external_depth:
            _plain_loss = _external_depth["plain-loss"]
            _ratio = _plain_loss["ratio"]
            _softness = _plain_loss.get("softness")
            if _softness is None:
              total_loss += ((original_depth - external_depth) ** 2).mean() * _ratio
            else:
              _patch_size = _softness.get("patch-size", 22)
              _samples = _softness.get("samples", 50)
              _margin = _softness.get("margin", 1e-4)
              depth_patches = original_depth.unfold(1, _patch_size, _patch_size).unfold(
                2,
                _patch_size,
                _patch_size,
              )
              external_depth_patches = (
                external_depth[None, ...]
                .unfold(1, _patch_size, _patch_size)
                .unfold(
                  2,
                  _patch_size,
                  _patch_size,
                )
              )
              _patches = depth_patches.size(1) * depth_patches.size(2)
              _patch_height_index = (
                torch.arange(depth_patches.size(1), device=external_depth_patches.device)
                .repeat_interleave(depth_patches.size(2))
                .repeat_interleave(_samples)
              )
              _patch_width_index = (
                torch.arange(depth_patches.size(2), device=external_depth_patches.device)
                .repeat(depth_patches.size(1))
                .repeat_interleave(_samples)
              )
              _sampled_pairs_x = torch.randint(
                low=0,
                high=_patch_size,
                size=(2, _samples * _patches),
                device=external_depth_patches.device,
              )
              _sampled_pairs_y = torch.randint(
                low=0,
                high=_patch_size,
                size=(2, _samples * _patches),
                device=external_depth_patches.device,
              )
              sampled_depth = depth_patches[
                0,
                _patch_height_index,
                _patch_width_index,
                _sampled_pairs_x,
                _sampled_pairs_y,
              ].transpose(0, 1)
              sampled_external_depth = external_depth_patches[
                0,
                _patch_height_index,
                _patch_width_index,
                _sampled_pairs_x,
                _sampled_pairs_y,
              ].transpose(0, 1)
              _lhs_larger = sampled_external_depth[:, 0] >= sampled_external_depth[:, 1]
              _lhs_lesser = ~_lhs_larger
              total_loss += (
                (
                  (sampled_depth[_lhs_larger][:, 1] - sampled_depth[_lhs_larger][:, 0] + _margin)
                  .clamp(min=0)
                  .sum()
                  + (sampled_depth[_lhs_lesser][:, 0] - sampled_depth[_lhs_lesser][:, 1] + _margin)
                  .clamp(min=0)
                  .sum()
                )
                / (_patches * _samples)
                * _ratio
              )

          if "statistics-loss" in _external_depth:
            _statistics_loss = _external_depth["statistics-loss"]
            _ratio = _statistics_loss["ratio"]
            _patch_size = _statistics_loss.get("patch-size", 28)
            _samples = _statistics_loss.get("samples")
            if _samples is None:
              original_statistics = full_statistics(original_depth, patch_size=_patch_size)
              external_statistics = camera.external_statistics
            else:
              original_statistics = full_statistics(
                original_depth,
                patch_size=_patch_size,
                samples=_samples,
              )
              external_statistics = full_statistics(
                external_depth[None, ...],
                patch_size=_patch_size,
                samples=original_statistics[3],
              )
            total_loss += _ratio * (
              ((original_statistics[0]["skewness"] - external_statistics[0]["skewness"]) ** 2).mean()
              + ((original_statistics[0]["kurtosis"] - external_statistics[0]["kurtosis"]) ** 2).mean()
              + ((original_statistics[1]["l_skewness"] - external_statistics[1]["l_skewness"]) ** 2).mean()
              + ((original_statistics[1]["l_kurtosis"] - external_statistics[1]["l_kurtosis"]) ** 2).mean()
              + ((original_statistics[2] - external_statistics[2]) ** 2).mean()
            )

        total_loss += loss

      return _ForwardPassResult(
        loss=total_loss,
        images=images,
        viewspace_point_tensors=viewspace_point_tensors,
        visibility_filters=visibility_filters,
        radiis=radiis,
        mean_depth_loss=sum(depth_losses) / len(depth_losses),
        mean_l1_loss=sum(l1_losses) / len(l1_losses),
      )

    with torch.no_grad():
      test_evaluate = _forward_pass(cams=scene.getTrainCameras())
      training_monitor.place_image(torch.cat(test_evaluate.images, dim=2))

    # first forward pass: get real rendering result and intermediate gradient
    first_pass_result = _forward_pass(cams=cams)

    # calculate the gradient
    first_pass_result.loss.backward()

    # collection information for densification
    with torch.no_grad():
      if not training_progress.densify_ended:
        for viewspace_point_tensor, visibility_filter, radii in zip(
          first_pass_result.viewspace_point_tensors,
          first_pass_result.visibility_filters,
          first_pass_result.radiis,
          strict=True,
        ):
          gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter],
            radii[visibility_filter],
          )
          gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        for cam, image in zip(cams, first_pass_result.images, strict=True):
          gt_image = cam.original_image.cuda()

          if "mvgs-like-w2w" in modifications or "mvgs-like-w2b" in modifications:
            loss_storage.append(torch.abs(gt_image - image).mean(dim=0))

          if any(mod in modifications for mod in ["projection", "multiplier"]):
            if cam not in ema_loss_history:
              ema_loss_history[cam] = torch.abs(image - gt_image).detach().clone()
              ema_loss_difference[cam] = torch.zeros_like(
                ema_loss_history[cam],
              )
            else:
              ema_loss_difference[cam] = (
                torch.abs(image - gt_image).detach().clone() - ema_loss_history[cam]
              ) * (1 - 0.9) + ema_loss_difference[cam] * 0.9
              ema_loss_history[cam] = torch.abs(image - gt_image).detach().clone()

          if "ldl" in modifications.get("metrics-to-trace", ()):
            current_loss[cam] = torch.abs(image - gt_image).detach().clone()
            if cam not in last_loss:
              last_loss[cam] = (current_loss[cam], training_progress.iteration)

          if "metrics-to-trace" in modifications:
            metrics_to_project = []
            if "l1" in modifications["metrics-to-trace"]:
              metrics_to_project.append(
                MetricProjectionSpecification(
                  name="l1",
                  lhs=torch.abs(image - gt_image).mean(dim=0),
                  rhs=None,
                  denom=None,
                ),
              )
            if "ssim" in modifications["metrics-to-trace"]:
              metrics_to_project.append(
                MetricProjectionSpecification(
                  name="ssim",
                  lhs=1 - ssim(image, gt_image, with_map=True)[1].mean(dim=0),
                  rhs=None,
                  denom=None,
                ),
              )
            if "frequency" in modifications["metrics-to-trace"]:
              metrics_to_project.append(
                MetricProjectionSpecification(
                  name="frequency",
                  lhs=cam.frequency,
                  rhs=None,
                  denom=None,
                ),
              )
            if "ldl" in modifications["metrics-to-trace"]:
              metrics_to_project.append(
                MetricProjectionSpecification(
                  name="ldl",
                  lhs=current_loss[cam]
                  * -torch.log(
                    1e-5 - (current_loss[cam] - last_loss[cam][0]).clamp(-1, 0),
                  ),
                  rhs=None,
                  denom=training_progress.iteration - last_loss[cam][1],
                ),
              )
            if "resnet" in modifications.get("metrics-to-trace", ()):
              gt_features = encode_without_resize(gt_image)
              rendered_features = encode_without_resize(image)
              difference = gt_features - rendered_features
              metrics_to_project.append(
                MetricProjectionSpecification(
                  name="resnet",
                  lhs=difference,
                  rhs=None,
                  denom=None,
                ),
              )
            gaussians.add_extra_densification_stats(camera=cam, metrics=tuple(metrics_to_project))

        if (
          any(mod in modifications for mod in ["projection", "multiplier"])
          and len(cams) > 1
          and cams[0].masks is not None
        ):
          # project loss back to gaussians
          for i, j in itertools.combinations(range(len(cams)), 2):
            lhs = cams[i]
            rhs = cams[j]

            # calculate metrics to project
            loss_difference = (
              (ema_loss_difference[lhs] * -1).clamp(1e-15, 1),
              (ema_loss_difference[rhs] * -1).clamp(1e-15, 1),
            )
            loss = ema_loss_history[lhs], ema_loss_history[rhs]
            metrics = (
              (loss[0] * -torch.log(loss_difference[0])).mean(dim=0),
              (loss[1] * -torch.log(loss_difference[1])).mean(dim=0),
            )

            for index in range(min(lhs.masks.max(), rhs.masks.max()) + 1):
              gaussians.project_metrics(
                bboxes=(lhs.bbox[index], rhs.bbox[index]),
                segments=(lhs.masks == index, rhs.masks == index),
                cameras=(lhs, rhs),
                metrics=metrics,
              )

    ## SAM grad implementation
    if sam_rho != 0:
      # collect parameters in gaussian model
      parameters = gaussians.parameters()
      # capture gradients
      original_grads = [parameter.grad.detach().clone() for parameter in parameters.values()]
      gradient_norm = torch.sqrt(sum(torch.sum(grad**2) for grad in original_grads))
      normalized_grads = [grad / (gradient_norm + 1e-12) for grad in original_grads]

      # backup parameters
      parameter_backup = [parameter.data.clone() for parameter in parameters]

      # update parameters with SAM
      for parameter, epsilon in zip(
        parameters.values(), (x * sam_rho for x in normalized_grads), strict=True
      ):
        parameter.data.add_(epsilon)

      # clear grad and recalculate the grad with updated parameters
      gaussians.optimizer.zero_grad(set_to_none=True)
      gaussians.exposure_optimizer.zero_grad(set_to_none=True)
      second_pass_result = _forward_pass(cams=cams)
      second_pass_result.loss.backward()

      # restore the parameters
      for parameter, backup in zip(parameters.values(), parameter_backup, strict=True):
        parameter.data.copy_(backup)

      # if we apply SAM gradients only to opacity, reset gradients back to the backed up values
      if sam_opacity_only:
        for (name, parameter), backup in zip(parameters.items(), original_grads, strict=True):
          if name == "opacity":
            continue
          parameter.grad.copy_(backup)

    iter_end.record()

    with torch.no_grad():
      # Progress bar
      ema_loss_for_log = 0.4 * first_pass_result.loss.item() + 0.6 * ema_loss_for_log
      ema_l_l1depth_for_log = 0.4 * first_pass_result.mean_depth_loss + 0.6 * ema_l_l1depth_for_log

      if training_progress.should_update_progress_bar:
        progress_bar.set_postfix(
          {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_l_l1depth_for_log:.{7}f}"},
        )

      # Log and save
      training_report(
        tb_writer=tb_writer,
        training_progress=training_progress,
        Ll1=first_pass_result.mean_l1_loss,
        loss=first_pass_result.loss,
        l1_loss=l1_loss,
        elapsed=iter_start.elapsed_time(iter_end),
        scene=scene,
        render_function=partial(
          render,
          pc=gaussians,
          pipe=pipe,
          bg_color=background,
          scaling_modifier=1.0,
          separate_sh=SPARSE_ADAM_AVAILABLE,
          override_color=None,
          use_trained_exp=dataset.train_test_exp,
          dropout_ratio=coad_dropout,
          dropout_by_opacity=True,
          opacity_noise=0.0,
        ),
        train_test_exp=dataset.train_test_exp,
      )
      if training_progress.should_save:
        print(f"\n[ITER {training_progress.iteration}] Saving Gaussians")
        # we scale down the opacity directly when storing the model so that the hyperparameter does not have
        #  to be passed to the renderer
        scene.save(
          training_progress.iteration,
          mappers=None if coad_dropout == 0.0 else {"opacity": lambda opacity: opacity * (1 - coad_dropout)},
        )

      # Densification
      if training_progress.should_attach_scale_prune_classifier:
        attach_classifier(
          target=gaussians,
          mode="prune",
          classifier="scale_based_prune",
          max_screen_size=20,
          extent=scene.cameras_extent,
        )

      if training_progress.should_densify:
        # do densification in this step
        if "new-projection" in modifications:
          for lhs, rhs in itertools.combinations(cams, 2):
            if (
              last_loss[lhs][1] == training_progress.iteration
              or last_loss[rhs][1] == training_progress.iteration
            ):
              continue
            lhs_metrics = current_loss[lhs] * -torch.log(
              1e-5 - (current_loss[lhs] - last_loss[lhs][0]).clamp(-1, 0),
            )
            rhs_metrics = current_loss[rhs] * -torch.log(
              1e-5 - (current_loss[rhs] - last_loss[rhs][0]).clamp(-1, 0),
            )
            for index in range(min(lhs.masks.max(), rhs.masks.max()) + 1):
              gaussians.project_metrics(
                bboxes=(lhs.bbox[index], rhs.bbox[index]),
                segments=(lhs.masks == index, rhs.masks == index),
                cameras=(lhs, rhs),
                metrics=(lhs_metrics.mean(dim=0), rhs_metrics.mean(dim=0)),
              )

        if "mvgs-like-w2w" in modifications or "mvgs-like-w2b" in modifications:
          # collect boxes
          if cams[0].masks is None:
            raise ValueError

          def _find_area(
            tag: int,
            masks: torch.Tensor,
            loss: torch.Tensor,
            bounding_box: torch.Tensor,
          ) -> tuple[torch.Tensor, float] | None:
            """Find a window on the image with greatest average loss."""
            _window_size = 20, 40

            if not bounding_box.any():
              return None

            target_box = None
            maximum_loss = None
            for i in range(bounding_box[0], bounding_box[2], _window_size[0]):
              for j in range(bounding_box[1], bounding_box[3], _window_size[1]):
                mask_slice = masks[i : i + _window_size[0], j : j + _window_size[1]]
                loss_slice = loss[i : i + _window_size[0], j : j + _window_size[1]]
                box = torch.as_tensor([i, j, i + mask_slice.shape[0] - 1, j + mask_slice.shape[1] - 1])
                box_size = mask_slice.shape[0] * mask_slice.shape[1]

                # most pixels in this area shall be tagged as expected tag
                if (mask_slice == tag).sum() < box_size * 0.75:
                  continue
                average_loss = loss_slice.mean()
                if maximum_loss is None or average_loss > maximum_loss:
                  maximum_loss = average_loss
                  target_box = box
            if target_box is None:
              return None
            return target_box, maximum_loss.item()

          _boxes = []
          for i, j in itertools.combinations(range(len(cams)), 2):
            lhs = cams[i]
            rhs = cams[j]

            for tag in range(min(lhs.masks.max(), rhs.masks.max()) + 1):
              lhs_result = _find_area(
                tag=tag,
                masks=lhs.masks,
                loss=loss_storage[i],
                bounding_box=lhs.bbox[tag],
              )
              rhs_result = _find_area(
                tag=tag,
                masks=rhs.masks,
                loss=loss_storage[j],
                bounding_box=rhs.bbox[tag],
              )
              if lhs_result is not None and rhs_result is not None:
                lhs_box, lhs_loss = lhs_result
                rhs_box, rhs_loss = rhs_result
                if "mvgs-like-w2w" in modifications:
                  _boxes.append(((i, lhs_box), (j, rhs_box), lhs_loss + rhs_loss))
                elif "mvgs-like-w2b" in modifications:
                  _boxes.append(((i, lhs_box), (j, rhs.bbox[tag]), lhs_loss))
                  _boxes.append(((i, lhs.bbox[tag]), (j, rhs_box), rhs_loss))
          _boxes.sort(key=lambda item: item[2], reverse=True)
          window_sources.box_pairs = [item[:2] for item in _boxes[:3]]
          window_sources.cameras = cams

        gaussians.densify_and_prune(extent=scene.cameras_extent)

        if training_progress.should_reset_opacity or (
          dataset.white_background and training_progress.at_densify_start_edge
        ):
          gaussians.reset_opacity()

      # Optimizer step
      gaussians.exposure_optimizer.step()
      gaussians.exposure_optimizer.zero_grad(set_to_none=True)
      if use_sparse_adam:
        visible = radii > 0
        gaussians.optimizer.step(visible, radii.shape[0])
      else:
        gaussians.optimizer.step()
      gaussians.optimizer.zero_grad(set_to_none=True)

      if training_progress.should_checkpoint:
        print(f"\n[ITER {training_progress.iteration}] Saving Checkpoint")
        _base_path = Path(scene.model_path) / "chkpnt"
        _base_path.mkdir(exist_ok=True)
        torch.save(
          (gaussians.capture(), training_progress.iteration),
          _base_path / f"{training_progress.iteration}.pt",
        )
  gaussians.densify_hook_manager.after_train(gaussian=gaussians)
  progress_bar.close()
  core_training_timer.save(target=dataset.model_path.joinpath("core_training_timing"))


def prepare_output_and_logger(args: Namespace) -> SummaryWriter | None:
  if args.model_path is None:
    unique_str = os.getenv("OAR_JOB_ID") if os.getenv("OAR_JOB_ID") else str(uuid.uuid4())
    args.model_path = Path.cwd().joinpath("output", unique_str[:10])

  # Set up output folder
  print(f"Output folder: {args.model_path}")
  args.model_path.mkdir(exist_ok=True)
  args.model_path.joinpath("cfg_args").write_text("\x00".join(sys.argv[1:]))
  with Path(args.model_path).joinpath("modifications").open("w") as modifications_f:
    print(modifications, file=modifications_f)

  # Create Tensorboard writer
  tb_writer = None
  if TENSORBOARD_FOUND:
    tb_writer = SummaryWriter(args.model_path)
  else:
    print("Tensorboard not available: not logging progress")
  return tb_writer


def training_report(
  tb_writer: SummaryWriter | None,
  training_progress: TrainingProgressManager,
  Ll1: torch.Tensor,
  loss: torch.Tensor,
  l1_loss: torch.Tensor,
  elapsed: float,
  scene: Scene,
  render_function: Callable[[Camera], dict[str, torch.Tensor]],
  *,
  train_test_exp: bool,
):
  if tb_writer:
    tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), training_progress.iteration)
    tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), training_progress.iteration)
    tb_writer.add_scalar("iter_time", elapsed, training_progress.iteration)

  # Report test and samples of training set
  if training_progress.should_test:
    torch.cuda.empty_cache()
    validation_configs = (
      {"name": "test", "cameras": scene.getTestCameras()},
      {
        "name": "train",
        "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
      },
    )

    for config in validation_configs:
      if config["cameras"] and len(config["cameras"]) > 0:
        l1_test = 0.0
        psnr_test = 0.0
        for idx, viewpoint in enumerate(config["cameras"]):
          image = torch.clamp(render_function(viewpoint_camera=viewpoint)["render"], 0.0, 1.0)
          gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
          if train_test_exp:
            image = image[..., image.shape[-1] // 2 :]
            gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
          if tb_writer and (idx < 5):
            tb_writer.add_images(
              config["name"] + f"_view_{viewpoint.image_name}/render",
              image[None],
              global_step=training_progress.iteration,
            )
            if training_progress.is_first_test_iteration:
              tb_writer.add_images(
                config["name"] + f"_view_{viewpoint.image_name}/ground_truth",
                gt_image[None],
                global_step=training_progress.iteration,
              )
          l1_test += l1_loss(image, gt_image).mean().double()
          psnr_test += psnr(image, gt_image).mean().double()
        psnr_test /= len(config["cameras"])
        l1_test /= len(config["cameras"])
        print(
          "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
            training_progress.iteration, config["name"], l1_test, psnr_test
          ),
        )
        if tb_writer:
          tb_writer.add_scalar(
            config["name"] + "/loss_viewpoint - l1_loss", l1_test, training_progress.iteration
          )
          tb_writer.add_scalar(
            config["name"] + "/loss_viewpoint - psnr", psnr_test, training_progress.iteration
          )

    if tb_writer:
      tb_writer.add_histogram(
        "scene/opacity_histogram", scene.gaussians.get_opacity, training_progress.iteration
      )
      tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], training_progress.iteration)
    torch.cuda.empty_cache()


if __name__ == "__main__":
  total_timer = Timer()
  for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).uuid)
  # Set up command line argument parser
  parser = ArgumentParser(description="Training script parameters")
  ModelParameters.install_parser(parser=parser)
  OptimizationParameters.install_parser(parser=parser)
  PipelineParameters.install_parser(parser=parser)
  TrainParameters.install_parser(parser=parser)
  prepare_modifications(parser)
  args = parser.parse_args()
  args.save_iterations.append(args.iterations)

  if "trace" in modifications and modifications["trace"].get("validate-capture", False):
    modifications["trace"]["capture"] = True

  if "debug" in modifications:
    export_traceback()

  print(f"Optimizing {args.model_path}")
  registers.register("out_path", Path(args.model_path))
  print(modifications)

  # Initialize system state (RNG)
  if "trace" in modifications and "capture" in modifications["trace"]:
    # if we are generating traces for second pass, do not fix the seed
    args.seed = randint(0, 2**32 - 1)  # noqa: S311 -- not for cryptographic usage
  safe_state(silent=args.quiet, seed=args.seed)

  torch.autograd.set_detect_anomaly(args.detect_anomaly)
  training(
    ModelParameters.from_parsed(parameters=args),
    OptimizationParameters.from_parsed(parameters=args),
    PipelineParameters.from_parsed(parameters=args),
    TrainParameters.from_parsed(parameters=args),
  )

  # All done
  print("\nTraining complete.")
  total_timer.save(target=args.model_path.joinpath("total_timing"))
