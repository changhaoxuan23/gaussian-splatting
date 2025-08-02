from __future__ import annotations

from argparse import ArgumentParser
from json import dump as json_dump
from pathlib import Path

import torch

from tracer.dataset import GaussianTraceDataset
from tracer.infinity_progress_bar import InfinityProgressBar
from tracer.network import GaussianPredictor


def _calculate_relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
  selector = target != 0
  prediction = prediction[selector]
  target = target[selector]
  return ((prediction - target).abs() / target.abs()).mean()


def _train(
  dataset: GaussianTraceDataset,
  network: GaussianPredictor,
  logging: Path | None,
  saving_path: Path,
) -> None:
  # optimizer
  optimizer = torch.optim.AdamW(network.parameters(), lr=0.003)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5)

  # data loader
  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, lengths=[0.8, 0.2])
  data = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=262144,
    shuffle=True,
    collate_fn=lambda x: x,
  )
  validation_data = validation_dataset.__getitems__(list(range(len(validation_dataset))))

  # progress control
  current_step = 0
  minimum_validation_loss = None

  # logging helpers
  bar = InfinityProgressBar()

  # train...
  total_log = []
  while True:
    for x, _y in data:
      y = _y * 1000
      current_step += 1
      optimizer.zero_grad()
      predicted_y = network(x)
      loss = torch.nn.functional.mse_loss(input=predicted_y, target=y)

      with torch.no_grad():
        relative_error = _calculate_relative_error(prediction=predicted_y, target=y)

        vx, _vy = validation_data
        vy = _vy * 1000
        predicted_vy = network(vx)
        relative_validation_error = _calculate_relative_error(prediction=predicted_vy, target=vy)
        validate_loss = torch.nn.functional.mse_loss(input=predicted_vy, target=vy)

      def _format_relative_error(value: float) -> str:
        if value >= 1:
          return str(int(value))
        return f"{value:.7f}"

      bar.config_information(name="step", value=current_step)
      bar.config_information(name="loss", value=f"{loss.item():.7f}")
      bar.config_information(name="v-loss", value=f"{validate_loss.item():.7f}")
      bar.config_information(name="r-error", value=_format_relative_error(relative_error.item()))
      bar.config_information(name="rv-error", value=_format_relative_error(relative_validation_error.item()))
      bar.config_information(name="lr", value=scheduler.get_last_lr()[0])

      if logging is not None:
        total_log.append(
          {
            "step": current_step,
            "l1": loss.item(),
            "vl1": validate_loss.item(),
            "r-error": relative_error.item(),
            "rv-error": relative_validation_error.item(),
            "lr": scheduler.get_last_lr()[0],
          },
        )
        with logging.open("w") as log_file:
          json_dump(total_log, log_file)

      loss.backward()
      optimizer.step()
      scheduler.step(validate_loss)

      if minimum_validation_loss is None or minimum_validation_loss > validate_loss:
        minimum_validation_loss = validate_loss
        network.dump(saving_path)
    if scheduler.get_last_lr()[0] < 1e-7:
      break


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--data-path", type=Path, required=True)
  parser.add_argument("--model-destination", type=Path, required=True)
  parser.add_argument("--log", type=Path)
  arguments = parser.parse_args()

  dataset = GaussianTraceDataset(data_path=arguments.data_path, device="cuda")
  network = GaussianPredictor(output_channels=dataset[0][1].shape[-1]).cuda()

  with torch.autograd.detect_anomaly():
    _train(
      dataset=dataset,
      network=network,
      logging=arguments.log,
      saving_path=arguments.model_destination,
    )
