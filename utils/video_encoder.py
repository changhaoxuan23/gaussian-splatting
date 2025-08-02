from pathlib import Path
from subprocess import DEVNULL, PIPE, Popen

import numpy as np
import torch


class VideoEncoder:
  def __init__(self, width: int, height: int, destination: Path, logging: Path | None) -> None:
    self._encoder = Popen(
      [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "1500",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx265",
        "-pix_fmt",
        "yuv420p",
        str(destination),
      ],
      stdin=PIPE,
      stdout=DEVNULL,
      stderr=DEVNULL if logging is None else logging.open("w"),
      close_fds=True,
    )
  
  def finalize(self) -> None:
    if self._encoder is not None:
      self._encoder.communicate()
      self._encoder = None

  def __del__(self) -> None:
    self.finalize()

  def place_image(self, image: torch.Tensor) -> None:
    image = image.permute(1, 2, 0)
    image = (image.clip(0, 1) * 255).to(torch.uint8).cpu().numpy()
    image = np.ascontiguousarray(image)
    self._encoder.stdin.write(image.tobytes())
