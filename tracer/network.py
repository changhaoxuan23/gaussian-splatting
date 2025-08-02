"""Network for predicting outcome of cloning/splitting certain gaussian.

Input format: N samples of 17 features
       step(1): percentage position of current step in densifying steps
   position(3): xyz coordinate of the gaussian
      scale(3): scale of the gaussian
   rotation(4): rotation of the gaussian
    opacity(1): opacity of the gaussian
       grad(1): norm of the gradient on position of the gaussian
        ldl(1): a new metric
         l1(1): l1 loss
       ssim(1): ssim loss
  frequency(1): frequency of corresponding image area

Output format: Nx3
   delta-ldl(1)
    delta-l1(1)
  delta-ssim(1)
"""

from pathlib import Path

import torch


def _build_embed_vector(target: int) -> torch.Tensor:
  index = torch.arange(target)
  return 2**index * torch.pi


class GaussianPredictor(torch.nn.Module):
  def __init__(
    self,
    output_channels: int,
    position_embedding_target: int = 10,
    step_embedding_target: int = 6,
  ) -> None:
    super().__init__()
    self._base = torch.nn.Sequential(
      torch.nn.Linear(
        in_features=position_embedding_target * 6 + step_embedding_target * 2 + 13,
        out_features=128,
      ),
      torch.nn.GELU(),
      torch.nn.Linear(in_features=128, out_features=256),
      torch.nn.GELU(),
      torch.nn.Linear(in_features=256, out_features=128),
      torch.nn.GELU(),
      torch.nn.Linear(in_features=128, out_features=128),
      torch.nn.GELU(),
      torch.nn.Linear(in_features=128, out_features=64),
      torch.nn.GELU(),
    )
    self._output_layers = torch.nn.ModuleList(
      modules=(
        torch.nn.Sequential(
          torch.nn.Linear(in_features=64, out_features=32),
          torch.nn.GELU(),
          torch.nn.Linear(in_features=32, out_features=1),
        )
        for _ in range(output_channels)
      ),
    )

    self.register_buffer(
      "_embedding_vector",
      _build_embed_vector(max(position_embedding_target, step_embedding_target)),
    )
    self._position_embedding_target = position_embedding_target
    self._step_embedding_target = step_embedding_target

  def _embed(self, x: torch.Tensor, target: int) -> torch.Tensor:
    vector = self._embedding_vector[:target]
    result = torch.empty(size=(len(x), target * 2 * x.shape[1]), dtype=x.dtype, device=x.device)
    source = vector.repeat(x.shape[1]) * x.repeat_interleave(target).view(len(x), target * x.shape[1])
    sine_part = torch.sin(source)
    cosine_part = torch.cos(source)
    result[:, 0::2] = sine_part
    result[:, 1::2] = cosine_part
    return result

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # preprocess input features: step and position need to be positional embedded
    steps = x[:, :1]
    positions = x[:, 1:4]
    others = x[:, 4:]

    embedded_steps = self._embed(steps, target=self._step_embedding_target)
    embedded_positions = self._embed(positions, target=self._position_embedding_target)

    x = torch.cat((embedded_steps, embedded_positions, others), dim=-1)

    base = self._base(x)
    return torch.cat(tuple(output_layer(base) for output_layer in self._output_layers), dim=-1)

  def dump(self, target: Path) -> None:
    torch.save(self.state_dict(), target)

  def load(self, source: Path) -> None:
    self.load_state_dict(torch.load(source, weights_only=True), strict=True)
