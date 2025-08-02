"""Helper functions in tracer evaluation."""

from typing import NamedTuple

import torch


class MetricDifferenceResult(NamedTuple):
  absolute_difference: torch.Tensor
  relative_difference: torch.Tensor
  effective_source_points: torch.Tensor


def calculate_metric_difference(
  source_points: torch.Tensor,  # (N, ): id of points selected by classifiers at last densification step
  source_metrics: torch.Tensor,  # (N, k): metrics for source points captured before last densification
  mapping: torch.Tensor,  # (M, s + 1): densify mapping: point [i, 0] have been densified into points [i, 1:]
  current_points: torch.Tensor,  # (W, ): id of points in gaussian model now
  current_metrics: torch.Tensor,  # (W, k): metrics of points now
) -> MetricDifferenceResult:
  """Calculate the difference between metrics captured before last densify and now.

  During each densify procedure, classifiers will select points to be densified in this step, which forms the
   source points. Each source point establishes a set of metrics, which forms the source metrics.
  At the moment this function is invoked, a densify_and_prune procedure have been carries out after the time
   source points and source metrics are captured. Points now are captured as current points, whose metrics
   are current metrics.
  The mapping maps source point to current point, but some of which may have been pruned.
  This function pairs source points with current points densified from them and calculate absolute and
   relative differences between their metrics.
  """
  # find source points that exists in both source_points and mapping
  #  the whole densification operation may include multiple mappings, while each mapping may include points
  #  selected by classifiers other than the predictor
  mapping_order = torch.argsort(mapping[:, 0])
  sorted_mapping_index = mapping[mapping_order][:, 0].contiguous()
  search_result = mapping[
    mapping_order[
      torch.searchsorted(sorted_mapping_index, source_points).clamp(min=None, max=len(mapping_order) - 1)
    ]
  ]
  effective_source_points = torch.ones(size=(len(source_points),), dtype=bool, device=source_points.device)
  mapped_points = search_result[:, 0] == source_points
  source_points = source_points[mapped_points]
  effective_source_points[torch.where(effective_source_points)[0][~mapped_points]] = False
  source_metrics = source_metrics[mapped_points]
  destination_points = search_result[mapped_points][:, 1:]  # (M, s): destinations of densify

  # find indices of destination points
  #  note that some destination points may be missing due to point pruning
  sorted_current_points, current_points_order = torch.sort(current_points)
  sorted_current_points = sorted_current_points.contiguous()
  search_result = torch.searchsorted(sorted_current_points, destination_points)
  search_result = search_result.clamp(min=None, max=len(current_points_order) - 1)
  destination_point_indices = current_points_order[search_result]  # (M, s)
  destination_point_mask = current_points[destination_point_indices] == destination_points
  effective_source_mask = destination_point_mask.sum(dim=-1) != 0
  # remove sources that have no destination left
  source_points = source_points[effective_source_mask]
  effective_source_points[torch.where(effective_source_points)[0][~effective_source_mask]] = False
  source_metrics = source_metrics[effective_source_mask]
  destination_points = destination_points[effective_source_mask]
  destination_point_indices = destination_point_indices[effective_source_mask]
  destination_point_mask = destination_point_mask[effective_source_mask]

  # calculate mean of current metrics
  current_metrics = current_metrics[destination_point_indices, :]
  current_metrics[~destination_point_mask] = 0
  current_mean = current_metrics.sum(dim=1) / destination_point_mask.sum(dim=-1)[:, None]

  # calculate difference
  difference = source_metrics - current_mean

  return MetricDifferenceResult(
    absolute_difference=difference,
    relative_difference=difference / (current_mean + 1e-8),
    effective_source_points=effective_source_points,
  )
