"""Several utilities for visualization."""
from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING, NamedTuple

import torch
from PIL import Image, ImageDraw, ImageFont
from visualize_tools.utils import fit_font

from gaussian_renderer import render
from tracer.gaussian_counter import count as count_gaussians

if TYPE_CHECKING:
  from collections.abc import Iterable

  from PIL.ImageDraw import _Ink

  from scene.cameras import Camera
  from scene.gaussian_protocols import RenderingGaussian


def clamp_extreme_values(
  data: torch.Tensor,
  minimum: float,
  maximum: float,
  *,
  cutting_point: float | None = None,
) -> torch.Tensor:
  flatted_data = data.view(-1).clone()
  if len(flatted_data) == 0:
    return flatted_data.reshape(data.shape)

  if cutting_point is not None:
    difference = flatted_data - cutting_point
    greater_selector = difference > 0
    lesser_selector = difference < 0
    difference[greater_selector] = clamp_extreme_values(
      data=difference[greater_selector],
      minimum=0,
      maximum=maximum,
    )
    difference[lesser_selector] = clamp_extreme_values(
      data=difference[lesser_selector],
      minimum=minimum,
      maximum=0,
    )
    return (difference + cutting_point).reshape(data.shape)
  sorted_data, order = torch.sort(flatted_data)
  if minimum > 0:
    cut_values = int(len(flatted_data) * minimum)
    flatted_data[order[:cut_values]] = sorted_data[cut_values]
  if maximum > 0:
    cut_values = int(len(flatted_data) * maximum)
    flatted_data[order[-cut_values:]] = sorted_data[-cut_values - 1]
  return flatted_data.reshape(data.shape)

class MaskedGaussianModelView:
  """Helper class that exports only masked points in the underlying gaussian model.

  This model can be helpful in case you want to render a subset of gaussians in a gaussian model without
   cloning or modifying it.
  """

  def __init__(self, gaussian: RenderingGaussian, mask: torch.Tensor) -> None:
    self._gaussian = gaussian
    self._mask = mask

    self.active_sh_degree = gaussian.active_sh_degree

  @property
  def get_scaling(self) -> torch.Tensor:
    return self._gaussian.get_scaling[self._mask]

  @property
  def get_rotation(self) -> torch.Tensor:
    return self._gaussian.get_rotation[self._mask]

  @property
  def get_xyz(self) -> torch.Tensor:
    return self._gaussian.get_xyz[self._mask]

  @property
  def get_features(self) -> torch.Tensor:
    return self._gaussian.get_features[self._mask]

  @property
  def get_features_dc(self) -> torch.Tensor:
    return self._gaussian.get_features_dc[self._mask]

  @property
  def get_features_rest(self) -> torch.Tensor:
    return self._gaussian.get_features_rest[self._mask]

  @property
  def get_opacity(self) -> torch.Tensor:
    return self._gaussian.get_opacity[self._mask]

  def get_covariance(self, scaling_modifier: float = 1) -> torch.Tensor:
    return self._gaussian.get_covariance(scaling_modifier=scaling_modifier)[self._mask]


def visualize_gaussian_count(
  gaussian: RenderingGaussian,
  camera: Camera,
  selector: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
  _gaussian = gaussian if selector is None else MaskedGaussianModelView(gaussian=gaussian, mask=selector)
  _counting = count_gaussians(
    gaussian=_gaussian,
    camera=camera,
  )
  _minimum, _maximum = torch.aminmax(_counting)
  _normalized_counting = (_counting - _minimum) / (_maximum - _minimum)
  _counting_image = _normalized_counting[None, ...].expand(3, -1, -1)
  return _counting_image, (_minimum.item(), _maximum.item(), len(_gaussian.get_xyz))

class ValueVisualizationColorPack(NamedTuple):
  zero_level_color: torch.Tensor
  positive_color: torch.Tensor
  negative_color: torch.Tensor


class ValueSpecification(NamedTuple):
  value: torch.Tensor
  zero_level: float = 0
  clamp: tuple[float, float] | None = (0.02, 0.02)
  apply_cutting_point: bool = True


class ValueVisualizationReturnPack(NamedTuple):
  images: Iterable[torch.Tensor]
  clamped_maximum: float
  raw_maximum: float
  clamped_minimum: float
  raw_minimum: float


CommonZeroLevelColor = torch.zeros(size=(3,), device="cuda")
CommonGoodColor = torch.as_tensor((0, 188, 212), device="cuda") / 255
CommonBadColor = torch.as_tensor((255, 87, 34), device="cuda") / 255
CommonValueVisualizationColorPack = ValueVisualizationColorPack(
  zero_level_color=CommonZeroLevelColor,
  positive_color=CommonGoodColor,
  negative_color=CommonBadColor,
)

_pipe_configuration = Namespace(
  debug=False,
  antialiasing=True,
  compute_cov3D_python=False,
  convert_SHs_python=False,
)


def visualize_value_on_gaussian(
  gaussian: RenderingGaussian,
  cameras: Iterable[Camera],
  value: ValueSpecification,
  colors: ValueVisualizationColorPack = CommonValueVisualizationColorPack,
) -> ValueVisualizationReturnPack:
  _value_unclamped = value.value.flatten() - value.zero_level
  if len(_value_unclamped) != len(gaussian.get_xyz):
    raise ValueError
  if value.clamp is not None:
    minimum, maximum = value.clamp
    _value = clamp_extreme_values(
      data=_value_unclamped,
      minimum=minimum,
      maximum=maximum,
      cutting_point=0 if value.apply_cutting_point else None,
    )

  color_override = torch.empty(size=(len(gaussian.get_xyz), 3), device="cuda")
  color_override[_value == 0] = colors.zero_level_color.cuda()
  color_override[_value > 0] = _value[_value > 0, None] / _value.max() * colors.positive_color.cuda()
  color_override[_value < 0] = _value[_value < 0, None] / _value.min() * colors.negative_color.cuda()

  return ValueVisualizationReturnPack(
    images=(
      render(
        viewpoint_camera=camera,
        pc=gaussian,
        pipe=_pipe_configuration,
        bg_color=colors.zero_level_color.cuda(),
        override_color=color_override,
      )["render"]
      for camera in cameras
    ),
    clamped_maximum=_value.max().item() + value.zero_level,
    raw_maximum=_value_unclamped.max().item() + value.zero_level,
    clamped_minimum=_value.min().item() + value.zero_level,
    raw_minimum=_value_unclamped.min().item() + value.zero_level,
  )

class VisualizeDistribution:
  """We use this as a namespace in C++."""

  def __init__(self) -> None:
    raise NotImplementedError

  class DataSpecification(NamedTuple):
    """Specify the data to be visualized."""

    data: torch.Tensor

    # ranges that each block represents. This should be an one-dimensional tensor with N + 1 elements, where
    #  N is the number of blocks to use. i-th block represents number of data instances within the range
    #  (ranges[i], ranges[i+1]], except for the 0-th block, where the range would be [ranges[i], ranges[i+1]]
    ranges: torch.Tensor

  class SizeSpecification(NamedTuple):
    """Specify the size in visualization.

    Specify size of things that we will fill by certain color here
    """

    # size of each block
    block: int

    # width of the range limit mark
    limit_mark: int

  class ColorSpecification(NamedTuple):
    """Specify colors used in the visualization.

    Each one must be a Tensor with shape == (3, ) in range [0, 1]
    """

    background: torch.Tensor
    maximum_value: torch.Tensor
    limit_mark: torch.Tensor

  CommonMaximumColors = (
    torch.as_tensor([156, 204, 101]) / 255,
    torch.as_tensor([41, 182, 246]) / 255,
  )
  CommonLimitColor = torch.as_tensor([77, 182, 172]) / 255

  @staticmethod
  def _make_line_smoother(original: torch.Tensor) -> torch.Tensor:
    _, block_size, total_width = original.shape
    data_columns = total_width // block_size
    _transform_length = max(round(block_size * 0.2), 1)
    _factors = torch.linspace(0, 1, _transform_length * 2 + 2, device=original.device)[1:-1]
    for i in range(1, data_columns):
      original[..., i * block_size - _transform_length : i * block_size + _transform_length] = (
        original[..., i * block_size - _transform_length - 1][..., None] * (1 - _factors)
        + original[..., i * block_size + _transform_length][..., None] * _factors
      )
    return original

  @staticmethod
  def visualize(
    data: DataSpecification,
    size: SizeSpecification,
    color: ColorSpecification,
    device: torch.device | str = "cpu",
  ) -> torch.Tensor:
    """Visualize the distribution of a set of data.

    This function will transform a set of data into one line of blocks, each representing, by its color, the
    number of data instances in the corresponding range
    """
    # load data to device
    background = color.background.to(device)
    maximum_value = color.maximum_value.to(device)
    limit_mark = color.limit_mark.to(device)
    ranges = data.ranges.to(device)
    values = data.data.to(device)

    # count instances in each block
    counting = torch.empty(size=(len(ranges) - 1,), device=device)
    #  selected (counted) data instances
    selected = torch.zeros(size=(len(values),), dtype=bool, device=device)
    #  start and end point: id of the first and last non-empty block
    start_point, end_point = None, None
    for i in range(len(data.ranges) - 1):
      selection = values <= ranges[i + 1]
      data_count = selection.logical_and(~selected).sum()
      if data_count != 0:
        if start_point is None:
          start_point = i
        end_point = i + 1
      counting[i] = data_count
      selected = selection

    # render blocks
    counting = (counting - counting.min()) / (counting.max() - counting.min())
    result = VisualizeDistribution._make_line_smoother(
      ((maximum_value - background)[:, None] * counting.repeat_interleave(size.block) + background[:, None])[
        :,
        None,
        :,
      ].repeat(1, size.block, 1),
    )

    # place endpoint mark
    result[..., start_point * size.block : start_point * size.block + size.limit_mark] = limit_mark[
      :,
      None,
      None,
    ]
    result[..., end_point * size.block - size.limit_mark : end_point * size.block] = limit_mark[:, None, None]

    return result


def add_title(
  image: Image.Image,
  text: str,
  height: int,
  font: ImageFont.FreeTypeFont,
  color: tuple[_Ink, _Ink],
) -> Image.Image:
  """Add a title to the image.

  This is done by adding some extra height to the top of input image then rendering the text in it.
  """
  result_image = Image.new(mode=image.mode, size=(image.width, image.height + height), color=color[0])
  result_image.paste(image, box=(0, height))
  font, _, _ = fit_font(
    texts=(text,),
    font=font,
    height=height,
    width=image.width,
  )
  draw = ImageDraw.Draw(result_image)
  draw.text(
    xy=(image.width // 2, 0),
    text=text,
    fill=color[1],
    font=font,
    anchor="mt",
  )
  return result_image
