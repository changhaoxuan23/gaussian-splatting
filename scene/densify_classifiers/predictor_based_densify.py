"""Densify based on predictor output."""

from __future__ import annotations

import json
from itertools import chain
from random import sample
from typing import TYPE_CHECKING, override

import torch

from scene.densification_classifiers_typing import GaussianADCClassifier
from tracer.evaluation_helper import calculate_metric_difference
from tracer.network import GaussianPredictor
from utils.adc_visualizer import register_gaussian_counting
from utils.modifications import modifications
from utils.visualize_utils import (
  CommonBadColor,
  CommonGoodColor,
  CommonZeroLevelColor,
  ValueSpecification,
  ValueVisualizationColorPack,
  visualize_gaussian_count,
  visualize_value_on_gaussian,
)

if TYPE_CHECKING:
  from pathlib import Path

  from scene.gaussian_protocols import TrainingGaussian
  from utils.adc_visualizer import ADCVisualizer
  from utils.train_progress_manager import TrainingProgressManager


def _visualize(
  visualizer: ADCVisualizer,
  predictions: tuple[tuple[str, torch.Tensor], ...],
  gaussians: TrainingGaussian,
  selections: torch.Tensor | None,
) -> None:
  for metric_name, value in predictions:
    visualization = visualize_value_on_gaussian(
      gaussian=gaussians,
      cameras=(adc_view.view for adc_view in visualizer.views),
      value=ValueSpecification(
        value=value,
        zero_level=0,
        clamp=(0.01, 0.01),
        apply_cutting_point=True,
      ),
      colors=ValueVisualizationColorPack(
        zero_level_color=CommonZeroLevelColor,
        positive_color=CommonBadColor,
        negative_color=CommonGoodColor,
      ),
    )
    for adc_view, image in zip(visualizer.views, visualization.images, strict=True):
      adc_view.register_visualization(
        annotation=(
          f"predicted {metric_name}",
          f"L={visualization.clamped_minimum}",
          f"RL={visualization.raw_minimum}",
          f"H={visualization.clamped_maximum}",
          f"RH={visualization.raw_maximum}",
        ),
        image=image,
      )

  if selections is not None:
    for adc_view in visualizer.views:
      register_gaussian_counting(
        adc_visualize_view=adc_view,
        counting_result=visualize_gaussian_count(
          gaussian=gaussians,
          camera=adc_view.view,
          selector=selections,
        ),
        annotation_prefixes=("predictor proposed",),
      )


class _PredictorBasedDensifyClassifier(GaussianADCClassifier):
  def __init__(
    self,
    thresholds: tuple[float, ...],
    training_progress: TrainingProgressManager,
    output_directory: Path,
  ) -> None:
    super().__init__()

    self._thresholds = thresholds
    self._training_progress = training_progress
    self._output_directory = output_directory

    self._last_predictor_selection: torch.Tensor | None = None
    self.last_predictor_prediction: torch.Tensor | None = None

    self._predictor = GaussianPredictor(output_channels=len(thresholds))
    self._predictor.load(source=modifications["trace"]["model"])
    self._predictor = self._predictor.cuda()

  def _evaluation_checker(
    self,
    gaussians: TrainingGaussian,
    current_values: tuple[torch.Tensor, ...],
  ) -> None:
    _selected_point_ids = self._last_predictor_selection.cpu().tolist()
    _selected_point_predictions = self._last_predictor_prediction.cpu().tolist()
    all_selected_points = [
      (point_id, tuple(point_prediction))
      for point_id, point_prediction in zip(_selected_point_ids, _selected_point_predictions, strict=True)
    ]

    _current_point_ids = gaussians.point_id.cpu().tolist()
    _current_point_metrics = torch.stack(current_values, dim=-1).cpu().tolist()
    all_current_points = [
      (point_id, tuple(point_metric))
      for point_id, point_metric in zip(_current_point_ids, _current_point_metrics, strict=True)
    ]

    _current_points = set(_current_point_ids)
    _split_mapping = gaussians.registers["split-mapping"].cpu().tolist()
    _clone_mapping = gaussians.registers["clone-mapping"].cpu().tolist()
    selected_point_mappings = [
      (entry[0], tuple(candidate for candidate in entry[1:] if candidate in _current_points))
      for entry in chain(_split_mapping, _clone_mapping)
    ]

    output_directory = self._output_directory.joinpath("predictor-evaluation-debug")
    output_directory.mkdir(exist_ok=True)
    with output_directory.joinpath(f"{self._training_progress.iteration:05d}.json").open("w") as f:
      json.dump(
        {
          "last": all_selected_points,
          "current": all_current_points,
          "mapping": selected_point_mappings,
        },
        f,
        indent=2,
      )

  def _evaluate(
    self,
    gaussians: TrainingGaussian,
    metric_names: tuple[str],
    predictions: tuple[torch.Tensor, ...],
    current_values: tuple[torch.Tensor, ...],
    selection: torch.Tensor,
  ) -> None:
    if self._last_predictor_selection is not None:
      self._evaluation_checker(gaussians=gaussians, current_values=current_values)
      last_selection = self._last_predictor_selection
      last_predictions = self._last_predictor_prediction
      split_mapping = gaussians.registers["split-mapping"]
      clone_mapping = gaussians.registers["clone-mapping"]

      # we abuse this function to calculate what we need: the difference between metrics predicted and the
      #  metrics we actually got now by simply replacing source_metrics with prediction
      split_result = calculate_metric_difference(
        source_points=last_selection,
        source_metrics=last_predictions,
        mapping=split_mapping,
        current_points=gaussians.point_id,
        current_metrics=torch.stack(current_values, dim=-1),
      )
      split_difference, split_percentage_difference = (
        split_result.absolute_difference,
        split_result.relative_difference * 100,
      )
      clone_result = calculate_metric_difference(
        source_points=last_selection,
        source_metrics=last_predictions,
        mapping=clone_mapping,
        current_points=gaussians.point_id,
        current_metrics=torch.stack(current_values, dim=-1),
      )
      clone_difference, clone_percentage_difference = (
        clone_result.absolute_difference,
        clone_result.relative_difference * 100,
      )
      difference = torch.cat((split_difference, clone_difference), dim=0)
      percentage_difference = torch.cat((split_percentage_difference, clone_percentage_difference), dim=0)

      target_path = self._output_directory.joinpath("predictor-evaluation")
      if "predictor_evaluation_cleaned" not in gaussians.registers:
        target_path.unlink(missing_ok=True)
        gaussians.registers["predictor_evaluation_cleaned"] = True
      with target_path.open("a") as f:
        data = {
          name: value.item()
          for name, value in zip(metric_names, difference.abs().mean(dim=0).cpu(), strict=True)
        }
        data.update(
          {
            f"{name}%": value.item()
            for name, value in zip(metric_names, percentage_difference.abs().mean(dim=0).cpu(), strict=True)
          },
        )
        json.dump(data, f)
        f.write("\n")

      target_directory = self._output_directory.joinpath("predictor-evaluation-details")
      target_directory.mkdir(exist_ok=True)
      torch.save(
        {name: value.cpu() for name, value in zip(metric_names, difference.permute(1, 0), strict=True)},
        target_directory.joinpath(f"{self._training_progress.iteration:05d}"),
      )

    self._last_predictor_selection = gaussians.point_id[selection]
    self._last_predictor_prediction = torch.stack(
      tuple(
        value + prediction * self._training_progress.densify_interval
        for value, prediction in zip(current_values, predictions, strict=True)
      ),
      dim=-1,
    )[selection]

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    n_gaussians = len(gaussians.get_xyz)
    capture = gaussians.capture_trace()
    direct_feature = capture[:-1]
    extra_metrics = capture[-1]
    main_features = torch.cat(
      (*direct_feature, *(features[:, None] for _, features in extra_metrics)),
      dim=-1,
    )
    _percentage_step = self._training_progress.finished_densify_ratio
    features = torch.cat(
      (
        torch.full(
          size=(len(main_features), 1),
          fill_value=_percentage_step,
          dtype=main_features.dtype,
          device=main_features.device,
        ),
        main_features,
      ),
      dim=-1,
    )
    predictions = self._predictor(features)
    _predict_selector = torch.as_tensor([False] * n_gaussians, device=gaussians.get_xyz.device)
    for i in range(len(self._thresholds)):
      _predict_selector[predictions[:, i] < self._thresholds[i]] = True

    dldl, dl1, dssim = predictions[:, 0], predictions[:, 1], predictions[:, 2]

    # decide densification limit
    _limit = modifications["trace"].get("step-limit", 0.0)
    if _limit == 0:
      limit = n_gaussians
    else:
      if _limit > 1:
        _limit = (_limit - 1) * _percentage_step + 1
        _limit = 2 ** (-_limit)
      limit = max(int(_limit * n_gaussians), 1)

    _n_predictor_selected = _predict_selector.sum()
    if visualizer is not None:
      _visualize(
        visualizer=visualizer,
        predictions=(("dldl", dldl), ("dl1", dl1), ("dssim", dssim)),
        gaussians=gaussians,
        selections=_predict_selector if _n_predictor_selected > limit else None,
      )
    if _n_predictor_selected > limit:
      exceeds = _n_predictor_selected - limit
      picker = torch.where(_predict_selector)[0]
      picked = picker[sample(range(len(picker)), k=exceeds)]
      _predict_selector[picked] = False
      print(
        f"Proposed {_n_predictor_selected} Gaussians, finally selected {_predict_selector.sum()} Gaussians",
      )

    if modifications["trace"].get("evaluate", False):
      self._evaluate(
        gaussians=gaussians,
        metric_names=("dldl", "dl1", "dssim"),
        predictions=(dldl, dl1, dssim),
        current_values=(
          gaussians.extra_metrics.ldl,
          gaussians.extra_metrics.l1,
          gaussians.extra_metrics.ssim,
        ),
        selection=_predict_selector,
      )

    output_directory = self._output_directory.joinpath("predictor_outputs")
    output_directory.mkdir(exist_ok=True)
    torch.save(
      {
        "dldl": dldl.cpu(),
        "dl1": dl1.cpu(),
        "dssim": dssim.cpu(),
      },
      output_directory.joinpath(f"{self._training_progress.iteration:05d}"),
    )

    return _predict_selector


Classifier = _PredictorBasedDensifyClassifier
