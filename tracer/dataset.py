from __future__ import annotations

from pathlib import Path

import torch


class GaussianTraceDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    data_path: Path,
    *,
    device: torch.device | str | None = None,
  ) -> None:
    super().__init__()

    self._data = torch.load(data_path).to(device)

  def __len__(self) -> int:
    return len(self._data)

  def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    return self._data[indices, :-3], self._data[indices, -3:]

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = self.__getitems__([index])
    return left.unsqueeze(0), right.unsqueeze(0)
