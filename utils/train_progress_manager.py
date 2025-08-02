"""Manager of training progress.

This module handles training progress:
  - holds counters for iterations, number of densifies, etc.
  - exports simple properties about if densify should be done in this iteration, etc.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import TYPE_CHECKING

from tqdm import tqdm

from utils.modifications import modifications

if TYPE_CHECKING:
  from collections.abc import Sequence

  from arguments import OptimizationParameters, TrainParameters


class TrainingProgressManager:
  def _prepare_for_camera_grouping(self) -> None:
    if not modifications.get("with-camera-grouping", False):
      return
    interval = None
    if isinstance(modifications["with-camera-grouping"], dict):
      interval = modifications["with-camera-grouping"].get("regroup-interval", None)
    self.should_regroup_camera = property(fget=lambda: self.iteration % (interval or self.iteration + 1) == 0)

  def __init__(
    self,
    start_iteration: int,
    optimization: OptimizationParameters,
    training: TrainParameters,
  ) -> None:
    self._optimization = optimization
    self._training = training

    # current iteration, starts with 1, therefore _iteration == 0 means the training progress has not started
    self._iteration = start_iteration
    # if scale based prune classifier has been attached
    self._scale_prune_classifier_attached = False

    self._progress_bar = tqdm(total=self.iterations_remaining, desc="Training progress")

    # all the stuff for modifications
    self._prepare_for_camera_grouping()
    self._external_depth_from = modifications.get("external-depth", {}).get("start-at", 0)

  @property
  def progress_bar(self) -> tqdm:
    return self._progress_bar

  @property
  def iteration(self) -> int:
    return self._iteration

  def step(self, *, n: int = 1) -> bool:
    return self.step_to(target=self.iteration + n)

  def step_to(self, *, target: int) -> bool:
    if target < 0:
      raise ValueError
    delta = target - self.iteration
    self._progress_bar.update(n=delta)
    self._iteration = target
    return self.iterations_remaining >= 0

  @property
  def iterations_remaining(self) -> int:
    return self._optimization.iterations - self.iteration

  @property
  def should_increase_sh_level(self) -> bool:
    return self.iteration % 1000 == 0

  @property
  def should_start_debug(self) -> bool:
    return self.iteration == self._training.debug_from + 1

  @property
  def should_update_progress_bar(self) -> bool:
    return self.iteration % 10 == 0

  @property
  def densify_interval(self) -> int:
    return self._optimization.densification_interval

  @property
  def densify_started(self) -> bool:
    return self.iteration > self._optimization.densify_from_iter

  @property
  def densify_ended(self) -> bool:
    return self.iteration >= self._optimization.densify_until_iter

  @property
  def in_densifying_period(self) -> bool:
    return self.densify_started and not self.densify_ended

  @property
  def at_densify_start_edge(self) -> bool:
    return self.iteration == self._optimization.densify_from_iter

  @property
  def should_densify(self) -> bool:
    return self.in_densifying_period and self.iteration % self.densify_interval == 0

  @property
  def first_densify_iteration(self) -> int:
    return (
      math.ceil((self._optimization.densify_from_iter + 1) / self.densify_interval) * self.densify_interval
    )

  @property
  def final_densify_iteration(self) -> int:
    return (self._optimization.densify_until_iter - 1) // self.densify_interval * self.densify_interval

  @property
  def densify_iterations(self) -> Sequence[int]:
    return range(
      self.first_densify_iteration,
      self.final_densify_iteration + 1,
      self.densify_interval,
    )

  @property
  def densify_iterations_done(self) -> Sequence[int]:
    """Iteration numbers of finished densification.

    This will never include current iteration.
    """
    _densify_iterations = self.densify_iterations
    return _densify_iterations[: bisect_left(_densify_iterations, self.iteration)]

  @property
  def densify_iterations_remaining(self) -> Sequence[int]:
    """Iteration numbers of remaining densification.

    This will never include current iteration.
    """
    _densify_iterations = self.densify_iterations
    return _densify_iterations[bisect_right(_densify_iterations, self.iteration) :]

  @property
  def total_densify_steps(self) -> int:
    return len(self.densify_iterations)

  @property
  def finished_densify_ratio(self) -> float:
    return self.densify_iterations_done / self.total_densify_steps

  @property
  def densify_steps_done(self) -> int:
    """Number of densification done.

    If self.should_densify returns true, current iteration is not counted as one step done.
    """
    return len(self.densify_iterations_done)

  @property
  def densify_steps_remaining(self) -> int:
    """Number of densification remaining.

    If self.should_densify returns true, current iteration is not counted as one step remaining.
    """
    return len(self.densify_iterations_remaining)

  @property
  def last_densify_iteration(self) -> int | None:
    """Iteration when last densification was performed.

    Current iteration will never be reported as result.
    """
    _finished_iterations = self.densify_iterations_done
    return _finished_iterations[-1] if _finished_iterations else None

  @property
  def next_densify_iteration(self) -> int | None:
    """Iteration to perform next densification.

    Current iteration will never be reported as result.
    """
    _remaining_iterations = self.densify_iterations_remaining
    return _remaining_iterations[0] if _remaining_iterations else None

  @property
  def should_reset_opacity(self) -> bool:
    return self.in_densifying_period and self.iteration % self._optimization.opacity_reset_interval == 0

  @property
  def should_save(self) -> bool:
    return self.iteration in self._training.save_iterations

  @property
  def should_checkpoint(self) -> bool:
    return self.iteration in self._training.checkpoint_iterations

  @property
  def should_test(self) -> bool:
    return self.iteration in self._training.test_iterations

  @property
  def is_first_test_iteration(self) -> bool:
    return self.iteration == self._training.test_iterations[0]

  @property
  def should_attach_scale_prune_classifier(self) -> bool:
    return (
      not self._scale_prune_classifier_attached
      and self.iteration == self._optimization.opacity_reset_interval + 1
    )

  @property
  def should_apply_external_depth(self) -> bool:
    return self.iteration >= self._external_depth_from
