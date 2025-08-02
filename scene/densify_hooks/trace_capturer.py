"""Hook that captures traces."""

from shutil import rmtree
from typing import override

import torch

from scene.densification_hook_typing import DensifyHook, PendingDensify
from scene.gaussian_protocols import TrainingGaussian
from utils.modifications import modifications
from utils.registers import registers


class TraceCapturer(DensifyHook):
  def __init__(self) -> None:
    self._event_type_map = {
      "shot": 0,
      "initialize": 0,
      "prune": 3,
      "clone": 1,
      "split": 2,
    }

    self._trace_directory = registers.out_path.joinpath("traces")
    rmtree(path=self._trace_directory, ignore_errors=True)
    self._trace_directory.mkdir(exist_ok=False)

    self._first_run = True

  def _save_trace(self, trace: dict) -> None:
    torch.save(
      {
        "step": registers.train_progress.iteration,
        "percentage_step": registers.train_progress.finished_densify_ratio,
        **trace,
      },
      self._trace_directory.joinpath(
        f"{self._event_type_map[trace['event_type']] + registers.train_progress.iteration * 10:06d}",
      ),
    )

  def _shot_metrics(self, gaussian: TrainingGaussian) -> None:
    position, scale, rotation, opacity, grad, extra_metrics = gaussian.capture_trace()
    self._save_trace(
      {
        "event_type": "shot",
        "points": gaussian.point_id.cpu(),
        "position": position.cpu(),
        "scale": scale.cpu(),
        "rotation": rotation.cpu(),
        "opacity": opacity.cpu(),
        "metrics": {
          "grad": grad.cpu(),
          **{name: value.cpu() for name, value in extra_metrics},
        },
      },
    )

  @override
  def before_densify_selection(self, gaussian: TrainingGaussian) -> None:
    if self._first_run:
      self._save_trace({"event_type": "initialize", "points": gaussian.point_id.cpu()})
      self._first_run = False

    self._shot_metrics(gaussian=gaussian)

  @override
  def after_each_densify_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None: ...

  @override
  def before_densify_applied(
    self,
    gaussian: TrainingGaussian,
    points: PendingDensify,
  ) -> None:
    self._save_trace(
      {
        "event_type": "clone",
        "from": gaussian.point_id[points.clone.selector].cpu(),
        "to": points.clone.new_gaussian.point_id.cpu(),
      },
    )
    self._save_trace(
      {
        "event_type": "split",
        "from": gaussian.point_id[points.split.selector].cpu(),
        "to": points.split.new_gaussian.point_id.cpu(),
      },
    )

  @override
  def before_prune_selection(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...

  @override
  def after_each_prune_classifier(
    self,
    gaussian: TrainingGaussian,
    name: str,
    selector: torch.Tensor,
  ) -> None: ...

  @override
  def before_prune_applied(
    self,
    gaussian: TrainingGaussian,
    selector: torch.Tensor,
  ) -> None:
    self._save_trace(
      {
        "event_type": "prune",
        "points": gaussian.point_id[selector].cpu(),
      },
    )

  @override
  def before_done(
    self,
    gaussian: TrainingGaussian,
  ) -> None: ...

  @override
  def after_train(
    self,
    gaussian: TrainingGaussian,
  ) -> None:
    self._shot_metrics(gaussian=gaussian)


def _should_launch() -> bool:
  return "trace" in modifications and modifications["trace"].get("capture", False)


HOOK = TraceCapturer
HOOK_REGISTER_CONDITION = _should_launch
