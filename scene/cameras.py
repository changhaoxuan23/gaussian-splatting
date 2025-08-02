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

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

from utils.general_utils import PILtoTorch
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.resnet_encoder import encode_image


def _nearest_neighbor_resample(source: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
  original_shape = source.shape
  if len(original_shape) < len(shape):
    raise ValueError
  target_shape = tuple(-1 for _ in range(len(original_shape) - len(shape))) + shape
  selector = tuple(
    slice(None)
    if target in (original, -1)
    else torch.linspace(start=0, end=original - 1, steps=target).long()
    for original, target in zip(original_shape, target_shape, strict=True)
  )

  return source[selector]


def _get_frequency(
  image: torch.Tensor,
  power_limit: float = 0.995,
) -> tuple[int, float]:
  _, height, width = image.shape
  frequency = torch.fft.fft2(image)
  coordinate = torch.sqrt(torch.fft.fftfreq(height)[:, None] ** 2 + torch.fft.fftfreq(width)[None, :] ** 2)
  frequency = torch.fft.fftshift(frequency, dim=(-2, -1))
  coordinate = torch.fft.fftshift(coordinate, dim=(-2, -1))
  power = frequency.abs() ** 2
  power = power.sum(dim=0)

  bands = 2 * max(height, width)
  bands_limits = torch.linspace(start=0, end=coordinate.max(), steps=bands)
  selectors = coordinate[None, ...] <= bands_limits[:, None, None]
  expected_power = power.sum() * power_limit
  for index, selector in enumerate(selectors):
    collected_power = power[selector].sum()
    if collected_power >= expected_power:
      return index, bands_limits[index].item()
  return bands, coordinate.max().item()

def _get_padded_size(size: Iterable[int]) -> tuple[int, ...]:
  return tuple((v + 13) // 14 * 14 for v in size)

def _pad_to_multiple(image: torch.Tensor) -> torch.Tensor:
  _, height, width = image.shape
  new_height, new_width = _get_padded_size(size=(height, width))
  print(new_height, new_width)
  return torchvision.transforms.functional.resize(image, size=(new_height, new_width))


class Camera(nn.Module):
  dino_model = None

  def __init__(
    self,
    resolution,
    colmap_id,
    R,
    T,
    FoVx,
    FoVy,
    depth_params,
    image,
    invdepthmap,
    image_name,
    uid,
    trans=np.array([0.0, 0.0, 0.0]),
    scale=1.0,
    data_device="cuda",
    train_test_exp=False,
    is_test_dataset=False,
    is_test_view=False,
    masks=None,
    frequency: torch.Tensor | Path | None = None,
    dino_feature: torch.Tensor | Path | None = None,
    resnet_feature: torch.Tensor | Path | None = None,
  ):
    super(Camera, self).__init__()

    self.uid = uid
    self.colmap_id = colmap_id
    self.R = R
    self.T = T
    self.FoVx = FoVx
    self.FoVy = FoVy
    self.image_name = image_name

    try:
      self.data_device = torch.device(data_device)
    except Exception as e:
      print(e)
      print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
      self.data_device = torch.device("cuda")

    resized_image_rgb = PILtoTorch(image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    self.alpha_mask = None
    if resized_image_rgb.shape[0] == 4:
      self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
    else:
      self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

    if train_test_exp and is_test_view:
      if is_test_dataset:
        self.alpha_mask[..., : self.alpha_mask.shape[-1] // 2] = 0
      else:
        self.alpha_mask[..., self.alpha_mask.shape[-1] // 2 :] = 0

    self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
    self.image_width = self.original_image.shape[2]
    self.image_height = self.original_image.shape[1]

    if masks is not None:
      masks = torch.as_tensor(masks).long().to(self.original_image.device)
      # resize the mask to align with image size
      h, w = self.original_image.shape[-2:]
      source_h, source_w = masks.shape
      ih = torch.linspace(0, source_h - 1, h).long().to(masks)
      iw = torch.linspace(0, source_w - 1, w).long().to(masks)
      masks = masks[ih[:, None], iw]
      # process segmentation information
      self.masks = masks
      _n_objects = masks.max() + 1
      self.bbox = torch.zeros(size=(_n_objects, 4), dtype=torch.long)
      for i in range(_n_objects):
        x, y = torch.where(self.masks == i)
        if len(y) == 0:
          continue
        self.bbox[i, 0] = torch.min(x)
        self.bbox[i, 1] = torch.min(y)
        self.bbox[i, 2] = torch.max(x)
        self.bbox[i, 3] = torch.max(y)
    else:
      self.masks = None
      self.bbox = None

    # prepare frequency of image
    if frequency is not None:
      if isinstance(frequency, Path):
        # we need to calculate the frequency
        _window_size = 40
        self.frequency = torch.empty_like(self.original_image[0])
        for i in range(0, self.image_height, _window_size):
          for j in range(0, self.image_width, _window_size):
            self.frequency[i : i + _window_size, j : j + _window_size] = _get_frequency(
              self.original_image[:, i : i + _window_size, j : j + _window_size],
            )[1]
        # save the frequency for subsequent usage
        frequency.parent.mkdir(exist_ok=True)
        torch.save(self.frequency.cpu(), frequency)
      else:
        # the frequency is loaded, just resize it to fit the resized image
        self.frequency = _nearest_neighbor_resample(
          source=frequency,
          shape=(self.image_height, self.image_width),
        ).to(self.original_image)
    
    # # prepare dino feature of image
    # if dino_feature is not None:
    #   if isinstance(dino_feature, Path):
    #     if Camera.dino_model is None:
    #       Camera.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    #       Camera.dino_model.eval()
    #     with torch.no_grad(), torch.autocast("cuda"):
    #       padded_size = _get_padded_size(gt_image.shape[1:])
    #       self.dino_feature = Camera.dino_model.get_intermediate_layers(_pad_to_multiple(gt_image)[None, ...])[0][0, ...].view(padded_size[0] // 14, padded_size[1] // 14, 768)
    #       dino_feature.parent.mkdir(exist_ok=True)
    #       torch.save(self.dino_feature.cpu(), dino_feature)
    #   else:
    #     self.dino_feature = dino_feature.to(self.original_image)
    #   self.dino_feature = torch.nn.functional.interpolate(self.dino_feature[None, ...].permute(0, 3, 1, 2), size=gt_image.shape[1:], mode="bilinear")[0, ...]
    
    # # prepare resnet feature of image
    # if resnet_feature is not None:
    #   if isinstance(resnet_feature, Path):
    #     self.resnet_feature = encode_image(gt_image[None, ...].cuda())
    #     resnet_feature.parent.mkdir(exist_ok=True)
    #     torch.save(self.resnet_feature.cpu(), resnet_feature)
    #   else:
    #     self.resnet_feature = resnet_feature.to(self.original_image)
    #   self.resnet_feature = torch.nn.functional.interpolate(self.resnet_feature, size=gt_image.shape[1:], mode="bilinear")[0, ...]
      

    self.invdepthmap = None
    self.depth_reliable = False
    if invdepthmap is not None:
      self.depth_mask = torch.ones_like(self.alpha_mask)
      self.invdepthmap = cv2.resize(invdepthmap, resolution)
      self.invdepthmap[self.invdepthmap < 0] = 0
      self.depth_reliable = True

      if depth_params is not None:
        if (
          depth_params["scale"] < 0.2 * depth_params["med_scale"]
          or depth_params["scale"] > 5 * depth_params["med_scale"]
        ):
          self.depth_reliable = False
          self.depth_mask *= 0

        if depth_params["scale"] > 0:
          self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

      if self.invdepthmap.ndim != 2:
        self.invdepthmap = self.invdepthmap[..., 0]
      self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

    self.zfar = 100.0
    self.znear = 0.01

    self.trans = trans
    self.scale = scale

    self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
    self.projection_matrix = (
      getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
      .transpose(0, 1)
      .cuda()
    )
    self.full_proj_transform = (
      self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

