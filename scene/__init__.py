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

import json
import os
import random
from collections.abc import Callable

import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from arguments import ModelParameters
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.graphics_utils import BasicPointCloud
from utils.modifications import modifications
from utils.registers import registers
from utils.system_utils import searchForMaxIteration


def farthest_point_sampling(arr, n_sample, start_idx=None):
  """Farthest Point Sampling without the need to compute all pairs of distance.

  Parameters
  ----------
  arr : numpy array
      The positional array of shape (n_points, n_dim)
  n_sample : int
      The number of points to sample.
  start_idx : int, optional
      If given, appoint the index of the starting point,
      otherwise randomly select a point as the start point.
      (default: None)

  Returns
  -------
  numpy array of shape (n_sample,)
      The sampled indices.

  Examples
  --------
  >>> import numpy as np
  >>> data = np.random.rand(100, 1024)
  >>> point_idx = farthest_point_sampling(data, 3)
  >>> print(point_idx)
      array([80, 79, 27])

  >>> point_idx = farthest_point_sampling(data, 5, 60)
  >>> print(point_idx)
      array([60, 39, 59, 21, 73])

  """
  n_points, n_dim = arr.shape

  if (start_idx is None) or (start_idx < 0):
    start_idx = random.randint(0, n_points)

  sampled_indices = [start_idx]
  min_distances = numpy.full(n_points, numpy.inf)

  for _ in range(n_sample - 1):
    current_point = arr[sampled_indices[-1]]
    dist_to_current_point = numpy.linalg.norm(arr - current_point, axis=1)
    min_distances = numpy.minimum(min_distances, dist_to_current_point)
    farthest_point_idx = numpy.argmax(min_distances)
    sampled_indices.append(farthest_point_idx)

  return numpy.array(sampled_indices)


class Arrow3D(FancyArrowPatch):
  def __init__(self, xs, ys, zs, *args, **kwargs):
    FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
    self._verts3d = xs, ys, zs

  def do_3d_projection(self, renderer=None):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

    return numpy.min(zs)


class Scene:
  gaussians: GaussianModel

  def __init__(
    self,
    args: ModelParameters,
    gaussians: GaussianModel,
    load_iteration=None,
    shuffle=True,
    resolution_scales=[1.0],
  ):
    """b
    :param path: Path to colmap scene main folder.
    """
    self.model_path = args.model_path
    self.loaded_iter = None
    self.gaussians = gaussians

    if load_iteration:
      if load_iteration == -1:
        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
      else:
        self.loaded_iter = load_iteration
      print("Loading trained model at iteration {}".format(self.loaded_iter))

    self.train_cameras = {}
    self.test_cameras = {}

    if os.path.exists(os.path.join(args.source_path, "sparse")):
      scene_info = sceneLoadTypeCallbacks["Colmap"](
        args.source_path,
        args.images,
        args.depths,
        args.evaluate,
        args.train_test_exp,
      )
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
      print("Found transforms_train.json file, assuming Blender data set!")
      scene_info = sceneLoadTypeCallbacks["Blender"](
        args.source_path,
        args.white_background,
        args.depths,
        args.evaluate,
      )
    else:
      assert False, "Could not recognize scene type!"

    if not self.loaded_iter:
      with (
        open(scene_info.ply_path, "rb") as src_file,
        open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file,
      ):
        dest_file.write(src_file.read())
      json_cams = []
      camlist = []
      if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
      if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
      for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
      with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
        json.dump(json_cams, file)

    if shuffle:
      random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
      random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

    self.cameras_extent = scene_info.nerf_normalization["radius"]
    registers.register(name="point_center", value=scene_info.nerf_normalization["translate"])
    registers.register(name="point_radius", value=scene_info.nerf_normalization["radius"])

    for resolution_scale in resolution_scales:
      print("Loading Training Cameras")
      self.train_cameras[resolution_scale] = cameraList_from_camInfos(
        scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False
      )
      print("Loading Test Cameras")
      self.test_cameras[resolution_scale] = cameraList_from_camInfos(
        scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True
      )

    if "with-camera-grouping" in modifications:
      self.train_camera_groups = {}
      self._n_regroups = 0
      self.rebuild_train_camera_groups()

    if self.loaded_iter:
      self.gaussians.load_ply(
        os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"),
        args.train_test_exp,
      )
    else:
      point_cloud = scene_info.point_cloud
      if "trace" in modifications and "capture" in modifications["trace"]:
        # if we are generating traces, append 20% extra random initialized points to the initial point cloud
        extra_points = int(len(point_cloud.points) * 0.2)

        generator = numpy.random.default_rng()
        # first generate random unit vectors
        directions = generator.random((extra_points, 3))
        directions = directions / numpy.linalg.norm(directions, axis=-1, keepdims=True)

        # then generate random length of offset
        _scale_modifier = 1.5
        scale = (
          generator.random((extra_points,)) * scene_info.nerf_normalization["radius"] * _scale_modifier * 2
          - scene_info.nerf_normalization["radius"] * _scale_modifier
        )[:, None]

        # we also apply a tiny perturbation to existing point cloud
        existing_points = len(point_cloud.points)
        perturbation = generator.random((existing_points, 3)) - 0.5
        perturbation = perturbation / numpy.linalg.norm(perturbation, axis=-1, keepdims=True)
        perturbation = (
          perturbation
          * generator.random((existing_points, 1))
          * scene_info.nerf_normalization["radius"]
          * 0.12
        )

        # make points
        point_cloud = BasicPointCloud(
          points=numpy.concatenate(
            (
              point_cloud.points + perturbation,
              directions * scale + scene_info.nerf_normalization["translate"],
            ),
            axis=0,
          ),
          colors=numpy.concatenate((point_cloud.colors, generator.random((extra_points, 3))), axis=0),
          normals=numpy.concatenate((point_cloud.normals, numpy.zeros((extra_points, 3))), axis=0),
        )
      self.gaussians.create_from_pcd(point_cloud, scene_info.train_cameras, self.cameras_extent)

  def save(
    self,
    iteration: int,
    mappers: dict[str, Callable[[np.typing.NDArray[np.float32]], np.float32]] | None = None,
  ):
    point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mappers=mappers)
    exposure_dict = {
      image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
      for image_name in self.gaussians.exposure_mapping
    }

    with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
      json.dump(exposure_dict, f, indent=2)

  def getTrainCameras(self, scale=1.0):
    return self.train_cameras[scale]

  def rebuild_train_camera_groups(self) -> None:
    print("Grouping Training Cameras")
    _n_groups = 16

    for resolution_scale in self.train_cameras:
      cameras = self.train_cameras[resolution_scale]
      camera_centers = numpy.stack(arrays=tuple(-camera.R.T @ camera.T for camera in cameras), axis=0)
      camera_directions = numpy.stack(
        arrays=tuple(camera.R.T @ numpy.array([[0.0], [0.0], [1.0]]) for camera in cameras),
        axis=0,
      )
      sampled_centers = farthest_point_sampling(camera_centers, _n_groups)

      distances = numpy.linalg.norm(
        camera_centers[None, ...] - camera_centers[sampled_centers, None, :],
        axis=-1,
      )
      leader_indices = numpy.argmin(distances, axis=0)
      camera_groups = []

      for index in range(_n_groups):
        indices = numpy.where(leader_indices == index)
        camera_groups.append(tuple(cameras[i] for i in indices[0]))
      self.train_camera_groups[resolution_scale] = camera_groups

      fig = plt.figure(figsize=(15, 15))
      ax = fig.add_subplot(111, projection="3d")
      colors = [(random.random(), random.random(), random.random()) for _ in range(_n_groups)]
      for camera_center, index in zip(camera_centers, leader_indices.tolist(), strict=True):
        ax.plot(
          [camera_center[0]],
          [camera_center[1]],
          [camera_center[2]],
          "o",
          markersize=10,
          color=colors[index],
        )
      for center, direction in zip(camera_centers, camera_directions, strict=True):
        d = direction / (numpy.linalg.norm(direction) + 1e-12) / 2
        arrow = Arrow3D(
          [center[0], center[0] + d[0][0]],
          [center[1], center[1] + d[1][0]],
          [center[2], center[2] + d[2][0]],
          mutation_scale=10,
          lw=0.6,
          arrowstyle="-|>",
          color="r",
        )
        ax.add_artist(arrow)
      registers.out_path.joinpath("camera-groupings").mkdir(exist_ok=True)
      plt.savefig(registers.out_path.joinpath("camera-groupings", f"{self._n_regroups:03d}.png"))

      if True:
        temporary_group_lists = [[] for _ in range(_n_groups)]
        for center, direction, index in zip(
          camera_centers,
          camera_directions,
          leader_indices.tolist(),
          strict=True,
        ):
          temporary_group_lists[index].append(
            {
              "center": center.tolist(),
              "direction": (direction / (numpy.linalg.norm(direction) + 1e-12)).tolist(),
            }
          )
        with registers.out_path.joinpath("cameras-reference.json").open("w") as fp:
          json.dump(temporary_group_lists, fp)
        exit()
    self._n_regroups += 1

  def getTrainCameraGroups(self, scale=1.0):
    return self.train_camera_groups[scale]

  def getTestCameras(self, scale=1.0):
    return self.test_cameras[scale]
