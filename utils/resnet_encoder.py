from typing import override

import torch
import torchvision
from PIL import Image

class _ResnetEncoderHelper:
  model = None
  picked_layers = (4, 5, 6)
  class _ResnetEncoder(torch.nn.Module):
    def __init__(self, model: torchvision.models.resnet.ResNet) -> None:
      super().__init__()

      self._layers = [m for m in model.children()][:max(_ResnetEncoderHelper.picked_layers) + 1]
    
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      results = []
      for (index, layer) in enumerate(self._layers):
        x = layer(x)
        if(index in _ResnetEncoderHelper.picked_layers):
          results.append(x)

      _reference_size = results[0].shape[2:]
      return torch.cat(tuple(
        torch.nn.functional.interpolate(result, size=_reference_size, mode="bilinear") for result in results
      ), dim=1)
  
  @staticmethod
  @torch.no_grad()
  def encode_image(image: torch.Tensor) -> torch.Tensor:
    if _ResnetEncoderHelper.model is None:
      base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=torchvision.models.ResNet18_Weights.DEFAULT)
      base_model = base_model.eval().cuda()
      _ResnetEncoderHelper.model = _ResnetEncoderHelper._ResnetEncoder(model=base_model)
    return _ResnetEncoderHelper.model(image)
  
  @staticmethod
  @torch.no_grad()
  def encode_without_resize(image: torch.Tensor) -> torch.Tensor:
    result = _ResnetEncoderHelper.encode_image(image[None, ...])
    return torch.nn.functional.interpolate(result, size=image.shape[1:], mode="bilinear")[0, ...]


encode_image = _ResnetEncoderHelper.encode_image
encode_without_resize = _ResnetEncoderHelper.encode_without_resize
  






  
  
