"""Visualizer for debugging and evaluating ADC procedure.

This utility provides a aligned view generator and a image container for visualization.
"""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import numpy
from PIL import Image, ImageDraw, ImageFont
from visualize_tools.utils import fit_font, tile_images

if TYPE_CHECKING:
  from collections.abc import Iterable, Sequence

  import torch
  from PIL.ImageDraw import _Ink
  from torch import Tensor

  from scene.cameras import Camera


def _place_lines(  # noqa: PLR0913
  image: Image.Image,
  lines: Sequence[str],
  font: ImageFont.FreeTypeFont,
  *,
  xy: tuple[int, int] = (0, 0),
  line_distance: int = 0,
  target_width: int | None = None,
  target_height: int | None = None,
  foreground: _Ink = "white",
  background: tuple[_Ink, float] | None = None,
  padding: int = 0,
) -> None:
  """Render lines to the image.

  Use xy to set top-left corner of the background box, default to the top-left corner of the image
  """
  drawer = ImageDraw.Draw(image)
  font, width, height = fit_font(
    texts=lines,
    font=font,
    width=target_width,
    height=target_height,
  )

  if background is not None:
    textbox_size = (width + padding * 2, height * len(lines) + line_distance * (len(lines) - 1) + padding * 2)
    background_box = Image.new(
      mode=image.mode,
      size=textbox_size,
      color=background[0],
    )
    image.paste(
      Image.blend(
        image.crop((*xy, xy[0] + textbox_size[0], xy[1] + textbox_size[1])),
        background_box,
        alpha=background[1],
      ),
      box=xy,
    )

  start_height = padding + xy[1]
  for line in lines:
    drawer.text(xy=(padding + xy[0], start_height), text=line, font=font, anchor="lt", fill=foreground)
    start_height += height + line_distance


_font = ImageFont.load_default()
_background = ("#546E7A", 0.4)
_line_distance = 2
_padding = 3


class ADCVisualizerViewHelper:
  def __init__(self, *, camera: Camera, register_target: list[Image.Image]) -> None:
    self._camera = camera
    self._target = register_target

  @property
  def view(self) -> Camera:
    return self._camera

  @property
  def name(self) -> str:
    return Path(self._camera.image_name).stem

  def register_visualization(self, *, annotation: Iterable[str], image: Tensor) -> None:
    _image = Image.fromarray((image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(numpy.uint8))
    _place_lines(
      image=_image,
      lines=annotation,
      font=_font,
      target_height=_image.height // 15,
      background=_background,
      padding=_padding,
      line_distance=_line_distance,
    )
    self._target.append(_image)


class ADCVisualizer:
  def __init__(self, *, seed: str, cameras: Iterable[Camera], views: int) -> None:
    self._generator = Random(seed)  # noqa: S311 -- not for cryptographic
    self._cameras = cameras
    self._views = views
    self._visualized_images: list[list[Image.Image]] = []

    self._step()

  def _step(self) -> None:
    self._current_views = self._generator.sample(self._cameras, k=self._views)
    self._visualized_images = [[] for _ in range(self._views)]

  def save(self, destination: Path) -> None:
    tile_images(
      images=tuple(tile_images(images=line, rows=1) for line in self._visualized_images),
      columns=1,
    ).save(destination)
    self._step()

  @property
  def views(self) -> Iterable[ADCVisualizerViewHelper]:
    return (
      ADCVisualizerViewHelper(camera=camera, register_target=self._visualized_images[i])
      for i, camera in enumerate(self._current_views)
    )

def register_gaussian_counting(
  adc_visualize_view: ADCVisualizerViewHelper,
  counting_result: tuple[torch.Tensor, tuple[int, int, int]],
  annotation_prefixes: Iterable[str],
) -> None:
  render_result, (minimum, maximum, total) = counting_result
  adc_visualize_view.register_visualization(
    annotation=(
      *annotation_prefixes,
      f"A={total}",
      f"G={maximum}",
      f"L={minimum}",
    ),
    image=render_result,
  )
