from pathlib import Path
from sys import argv

import numpy
import torch
from PIL import Image


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


input_path = Path(argv[1])
image = Image.open(input_path)
image = torch.as_tensor(numpy.array(image) / 255).permute(2, 0, 1)
for window_size in (5, 10, 20, 30, 40):
  frequency = torch.empty_like(image[0])
  for i in range(0, image.shape[1], window_size):
    for j in range(0, image.shape[2], window_size):
      frequency[i : i + window_size, j : j + window_size] = _get_frequency(
        image[:, i : i + window_size, j : j + window_size],
      )[1]
  frequency = (frequency - frequency.min()) / (frequency.max() - frequency.min())
  result_image = torch.zeros(size=(image.shape[0], image.shape[1] * 2, image.shape[2] * 2))
  result_image[:, :image.shape[1], :image.shape[2]] = image
  result_image[:, :image.shape[1], image.shape[2]:] = frequency[None, ...]
  result_image[:, image.shape[1]:, :image.shape[2]] = frequency[None, ...] * 0.6 + image * 0.4
  frequency_image = Image.fromarray((result_image.permute(1, 2, 0).numpy() * 255).astype(numpy.uint8))
  frequency_image.save(f"freq_{window_size:02d}.png")
