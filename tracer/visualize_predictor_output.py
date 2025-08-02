from itertools import cycle
from pathlib import Path
from sys import argv

import numpy
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont
from visualize_tools.utils import fit_font

_pixel_size = 9
_data_columns = 256
_anchor_length = 2
_annotation_space = 4
_border_width = 1
_colors = torch.as_tensor([156, 204, 101]) / 255, torch.as_tensor([41, 182, 246]) / 255
_zero_value_line = torch.as_tensor([255, 112, 67]) / 255
_line_range_color = torch.as_tensor([77, 182, 172]) / 255
_entries = "dldl", "dl1", "dssim"


def _make_line_smoother(original: torch.Tensor) -> torch.Tensor:
  _transform_length = max(round(_pixel_size * 0.2), 1)
  _factors = torch.linspace(0, 1, _transform_length * 2 + 2)[1:-1]
  for i in range(1, _data_columns):
    original[..., i * _pixel_size - _transform_length : i * _pixel_size + _transform_length] = (
      original[..., i * _pixel_size - _transform_length - 1][..., None] * (1 - _factors)
      + original[..., i * _pixel_size + _transform_length][..., None] * _factors
    )
  return original


path_list = sorted(Path(argv[1]).iterdir())

# collect ranges
_ranges = dict.fromkeys(_entries)
for path in path_list:
  data = torch.load(path)
  for entry in _entries:
    minimum, maximum = torch.aminmax(data[entry])
    _ranges[entry] = (
      (minimum, maximum)
      if _ranges[entry] is None
      else (min(_ranges[entry][0], minimum), max(_ranges[entry][1], maximum))
    )
_ranges = {
  entry: torch.linspace(minimum, maximum, _data_columns + 1) for entry, (minimum, maximum) in _ranges.items()
}

# build images
_data_image = {
  entry: torch.empty(size=(3, _pixel_size * len(path_list), _pixel_size * _data_columns))
  for entry in _entries
}
for _index, (path, color) in tqdm.tqdm(
  enumerate(zip(path_list, cycle(_colors), strict=False)),
  desc="building images",
  dynamic_ncols=True,
  total=len(path_list),
):
  _line = slice(_index * _pixel_size, _index * _pixel_size + _pixel_size)
  data = torch.load(path)
  for entry in _entries:
    _counting = torch.empty(size=(_data_columns,))
    _last_selection = None
    _data = data[entry]
    _limits = _ranges[entry][1:]
    _first_index, _last_index = None, 0
    for _column, _limit in enumerate(_limits):
      selection = _data <= _limit
      _pending_last_selection = selection
      if _last_selection is not None:
        selection = selection.logical_and(~_last_selection)
      current_counting = selection.sum()
      _counting[_column] = current_counting
      if current_counting != 0:
        _last_index = _column
        if _first_index is None:
          _first_index = _column
      _last_selection = _pending_last_selection
    _counting = (_counting - _counting.min()) / (_counting.max() - _counting.min())
    _data_image[entry][:, _line, :] = (color[:, None] * _counting.repeat_interleave(_pixel_size))[:, None, :]
    _make_line_smoother(_data_image[entry][:, _line, :])
    _data_image[entry][
      :,
      _line,
      [_first_index * _pixel_size, _last_index * _pixel_size + _pixel_size - 1],
    ] = _line_range_color[:, None, None]
    _data_image[entry][
      :,
      [
        _index * _pixel_size,
        _index * _pixel_size,
        _index * _pixel_size + _pixel_size - 1,
        _index * _pixel_size + _pixel_size - 1,
      ],
      [
        _first_index * _pixel_size + 1,
        _last_index * _pixel_size + _pixel_size - 2,
        _first_index * _pixel_size + 1,
        _last_index * _pixel_size + _pixel_size - 2,
      ],
    ] = _line_range_color[:, None]

# prepare font
font = ImageFont.load_default()
_texts = ["1", str(len(path_list))]
for entry in _entries:
  _texts.extend(
    [
      str(_ranges[entry][0].item()),
      str(_ranges[entry][len(_ranges[entry]) // 2].item()),
      str(_ranges[entry][-1].item()),
    ],
  )
font, render_width, _ = fit_font(texts=_texts, font=font, height=_pixel_size)

# enlarge image
_spacings = (
  _annotation_space + _border_width,
  _annotation_space + _border_width,
  _annotation_space + _pixel_size + _annotation_space + _anchor_length + _border_width,
  _annotation_space + render_width + _annotation_space + _anchor_length + _border_width,
)  # top, right, bottom, left
_image_tensors = {
  entry: torch.zeros(
    size=(
      3,
      _pixel_size * len(path_list) + _spacings[0] + _spacings[2],
      _pixel_size * _data_columns + _spacings[1] + _spacings[3],
    ),
  )
  for entry in _entries
}
# copy data image, draw decorations
for entry in _entries:
  # copy data image
  _image_tensors[entry][:, _spacings[0] : -_spacings[2], _spacings[3] : -_spacings[1]] = _data_image[entry]

  # draw borders
  _image_tensors[entry][
    :,
    _spacings[0] - _border_width : _spacings[0],
    _spacings[3] - _border_width : _border_width - _spacings[1],
  ] = 1.0
  _image_tensors[entry][
    :,
    -_spacings[2] : _border_width - _spacings[2],
    _spacings[3] - _border_width : _border_width - _spacings[1],
  ] = 1.0

  _image_tensors[entry][
    :,
    _spacings[0] : -_spacings[2],
    _spacings[3] - _border_width : _spacings[3],
  ] = 1.0
  _image_tensors[entry][
    :,
    _spacings[0] : -_spacings[2],
    -_spacings[1] : _border_width - _spacings[1],
  ] = 1.0

  # draw anchors
  ## anchors on rows
  _image_tensors[entry][
    :,
    _spacings[0] + _pixel_size * 0 + _pixel_size // 2 : _spacings[0] + _pixel_size * 0 + _pixel_size // 2 + 1,
    _spacings[3] - _border_width - _anchor_length : _spacings[3] - _border_width,
  ] = 1.0
  _image_tensors[entry][
    :,
    _spacings[0] + _pixel_size * (len(path_list) - 1) + _pixel_size // 2 : _spacings[0]
    + _pixel_size * (len(path_list) - 1)
    + _pixel_size // 2
    + 1,
    _spacings[3] - _border_width - _anchor_length : _spacings[3] - _border_width,
  ] = 1.0
  ## anchors on columns
  _image_tensors[entry][
    :,
    _border_width - _spacings[2] : _border_width + _anchor_length - _spacings[2],
    _spacings[3] : _spacings[3] + 1,
  ] = 1.0
  _image_tensors[entry][
    :,
    _border_width - _spacings[2] : _border_width + _anchor_length - _spacings[2],
    -_spacings[1] - 1 : -_spacings[1],
  ] = 1.0
  _image_tensors[entry][
    :,
    _border_width - _spacings[2] : _border_width + _anchor_length - _spacings[2],
    _spacings[3] + _pixel_size * (_data_columns // 2) - 1 : _spacings[3]
    + _pixel_size * (_data_columns // 2)
    + 1,
  ] = 1.0
  ## the zero-value line
  if _ranges[entry][0] <= 0 and _ranges[entry][-1] >= 0:
    _offset = (-_ranges[entry][0]) / (_ranges[entry][-1] - _ranges[entry][0]) * _pixel_size * _data_columns
    _offset = int(_offset)
    _image_tensors[entry][
      :,
      _spacings[0] : -_spacings[2],
      _spacings[3] + _offset : _spacings[3] + _offset + 1,
    ] = _zero_value_line[:, None, None]

images = {
  entry: Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(numpy.uint8))
  for entry, tensor in _image_tensors.items()
}
# attach annotations
for entry, image in images.items():
  draw = ImageDraw.Draw(image)
  draw.text(
    xy=(_spacings[3] - _annotation_space - _anchor_length - _border_width, _spacings[0] + _pixel_size // 2),
    text="1",
    fill="white",
    font=font,
    anchor="rm",
  )
  draw.text(
    xy=(
      _spacings[3] - _annotation_space - _anchor_length - _border_width,
      _spacings[0] + _pixel_size * (len(path_list) - 1) + _pixel_size // 2,
    ),
    text=str(len(path_list)),
    fill="white",
    font=font,
    anchor="rm",
  )

  draw.text(
    xy=(
      _spacings[3],
      _spacings[0] + _pixel_size * len(path_list) + _border_width + _anchor_length + _annotation_space,
    ),
    text=f"{_ranges[entry][0].item()} (1 block = {(_ranges[entry][1] - _ranges[entry][0]).item()})",
    fill="white",
    font=font,
    anchor="lt",
  )
  draw.text(
    xy=(
      _spacings[3] + _pixel_size * _data_columns // 2,
      _spacings[0] + _pixel_size * len(path_list) + _border_width + _anchor_length + _annotation_space,
    ),
    text=str(_ranges[entry][len(_ranges[entry]) // 2].item()),
    fill="white",
    font=font,
    anchor="mt",
  )
  draw.text(
    xy=(
      _spacings[3] + _pixel_size * _data_columns,
      _spacings[0] + _pixel_size * len(path_list) + _border_width + _anchor_length + _annotation_space,
    ),
    text=str(_ranges[entry][-1].item()),
    fill="white",
    font=font,
    anchor="rt",
  )

_output_directory = Path(argv[2])
_output_directory.mkdir(exist_ok=True)
for entry, image in images.items():
  image.save(_output_directory.joinpath(f"{entry}.png"))
