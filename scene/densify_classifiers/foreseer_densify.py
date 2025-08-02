"""Densify as if you are a foreseer."""

from __future__ import annotations

import math
from argparse import Namespace
from copy import deepcopy
from functools import partial
from itertools import chain
from random import sample
from shutil import rmtree
from typing import TYPE_CHECKING, NamedTuple, override

import numpy
import torch
from cuml import DBSCAN
from PIL import Image, ImageFont, ImageOps
from visualize_tools.utils import tile_images

from gaussian_renderer import render
from scene.densification_classifiers_typing import GaussianADCClassifier
from scene.densify_classifiers.grad_based_densify import Classifier as GradBasedDensify
from scene.densify_classifiers.random_densify import Classifier as RandomDensify
from tracer.evaluation_helper import calculate_metric_difference
from utils.adc_visualizer import ADCVisualizer
from utils.loss_utils import ssim
from utils.visualize_utils import (
  CommonBadColor,
  CommonGoodColor,
  CommonZeroLevelColor,
  MaskedGaussianModelView,
  ValueSpecification,
  ValueVisualizationColorPack,
  VisualizeDistribution,
  add_title,
  clamp_extreme_values,
  visualize_value_on_gaussian,
)

if TYPE_CHECKING:
  from collections.abc import Iterable, Sequence
  from pathlib import Path

  from arguments import OptimizationParameters
  from scene.cameras import Camera
  from scene.gaussian_protocols import TrainingGaussian
  from utils.train_progress_manager import TrainingProgressManager

def _entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> float:
  """Compute entropy in bits for a 1D tensor of probabilities."""
  probs = probs.clamp(min=eps)
  return -float(torch.sum(probs * torch.log2(probs)))


def _compute_nmi(
  clustering_result: torch.LongTensor,
  positive_gaussians: torch.Tensor,
  negative_gaussians: torch.Tensor,
  alpha: float = 1.0,  # Laplace smoothing
  ignore_label: int = -1,
) -> dict[str, object]:
  # Build mask of labeled samples (either positive or negative)
  labeled_mask = positive_gaussians | negative_gaussians
  # Exclude ignored cluster entries
  valid_mask = labeled_mask & (clustering_result != ignore_label)

  # If nothing to compute
  total_labeled = int(valid_mask.sum().item())
  if total_labeled == 0:
    return {
      "H_Y": 1,
      "H_C": 1,
      "H_Y_given_C": 1,
      "I_YC": 0,
      "NMI": 0,
      "per_cluster": {},
      "total_labeled": 0,
    }

  # Gather cluster ids of labeled valid samples
  cluster_ids = clustering_result[valid_mask]
  labels_pos = positive_gaussians[valid_mask]
  labels_neg = negative_gaussians[valid_mask]
  # Sanity: label should be exactly pos xor neg
  if not torch.all(labels_pos ^ labels_neg):
    # If some samples neither pos nor neg, treat them as unlabeled (shouldn't happen)
    raise ValueError

  # Map cluster ids to a contiguous index if they are arbitrary integers
  unique_clusters, inverse_idx = torch.unique(cluster_ids, sorted=True, return_inverse=True)
  k = unique_clusters.numel()

  # Count per-cluster totals and per-label counts
  nj = torch.zeros(k, dtype=torch.long)  # counts per cluster
  n_j_1 = torch.zeros(k, dtype=torch.long)  # counts label=1 (positive)
  # label=0 count = nj - n_j_1

  for idx in range(k):
    mask_cluster = inverse_idx == idx
    cnt = int(mask_cluster.sum().item())
    nj[idx] = cnt
    n_j_1[idx] = int(labels_pos[mask_cluster].sum().item())

  n = int(nj.sum().item())  # should equal total_labeled

  # Label marginals
  total_pos = int(n_j_1.sum().item())
  total_neg = n - total_pos

  # Compute H(Y)
  p_pos = (total_pos + 0.0) / n
  p_neg = (total_neg + 0.0) / n
  # if p_pos or p_neg are 0, entropy is 0
  probs_Y = torch.tensor([p_neg, p_pos], dtype=torch.float64)
  H_Y = _entropy_from_probs(probs_Y)

  # Compute H(C)
  # Use cluster marginals P(C=j) = nj / n
  p_C = nj.to(torch.float64) / float(n)
  H_C = _entropy_from_probs(p_C)

  # For each cluster compute smoothed conditional probs P(Y=1|C=j) with Laplace smoothing
  per_cluster = {}
  H_Y_given_C = 0.0
  eps = 1e-12
  for idx in range(k):
    count_j = float(nj[idx].item())
    if count_j == 0.0:
      # skip empty (shouldn't occur due to unique on present clusters)
      continue
    # smoothed counts
    smoothed_pos = float(n_j_1[idx].item()) + alpha
    smoothed_total = count_j + 2.0 * alpha
    p1_given_j = smoothed_pos / smoothed_total
    p0_given_j = 1.0 - p1_given_j
    probs_given_j = torch.tensor([p0_given_j, p1_given_j], dtype=torch.float64)
    H_given_j = _entropy_from_probs(probs_given_j)
    weight = count_j / n
    H_Y_given_C += weight * H_given_j

    per_cluster[int(unique_clusters[idx].item())] = {
      "n": int(count_j),
      "p0": p0_given_j,
      "p1": p1_given_j,
      "H_Y_given_Cj": H_given_j,
    }

  # Mutual information and NMI
  I_YC = H_Y - H_Y_given_C
  # Avoid division by zero for NMI; if H_Y or H_C == 0, set NMI to 0 (no information or degenerate)
  NMI = 0.0 if H_Y <= 0.0 or H_C <= 0.0 else I_YC / math.sqrt(max(H_Y * H_C, eps))

  return {
    "H_Y": H_Y,
    "H_C": H_C,
    "H_Y_given_C": H_Y_given_C,
    "I_YC": I_YC,
    "NMI": NMI,
    "per_cluster": per_cluster,
    "total_labeled": n,
  }


class _ForeseerDensifyClassifier(GaussianADCClassifier):
  @torch.no_grad()
  def _capture_gaussian_metrics(self, gaussian: TrainingGaussian) -> torch.Tensor:
    return torch.stack(
      tuple(getter(gaussian) for getter in self._metric_getter),
      dim=-1,
    )

  def __init__(
    self,
    optimization_config: OptimizationParameters,
    training_progress: TrainingProgressManager,
    *,
    use_gradient: bool,
    random_ratio: float,
    metrics_to_use: Iterable[str],
    strict: bool,
    target_gaussian: int,
    visualize_views: int,
    cameras: Sequence[Camera] | None,
    output_directory: Path | None,
    temperature_modifier: float,
    clustering_distance: float,
    neighbor_radius: float,
  ) -> None:
    super().__init__()

    self._training_progress = training_progress
    self._optimization_config = optimization_config

    # build exploring classifiers
    self._exploring_classifiers = []
    if use_gradient:
      self._exploring_classifiers.append(("grad", GradBasedDensify(optimization_config=optimization_config)))
    if random_ratio != 0:
      self._exploring_classifiers.append(("random", RandomDensify(ratio=random_ratio)))

    # build metric selector
    _metric_getter_mapper = {
      "dldl": lambda gaussian: gaussian.extra_metrics.ldl,
      "dl1": lambda gaussian: gaussian.extra_metrics.l1,
      "dssim": lambda gaussian: gaussian.extra_metrics.ssim,
    }
    self._metric_getter = [_metric_getter_mapper[metric] for metric in metrics_to_use]
    self._metrics_to_use = metrics_to_use
    if not self._metric_getter:
      raise ValueError

    # the reducer decides how to transform multiple metrics of a gaussian into a single value
    #  gaussian must have all its metrics decreased to be selected as a candidate, therefore we use minimum
    #   for reducer; maximum otherwise. See comment about difference in _ForeseerDensifyClassifier._foreseer.
    self._metric_reducer = lambda tensor: (torch.min if strict else torch.max)(tensor, dim=-1)[0]
    self._target_gaussian = target_gaussian

    # temperature control
    self._temperature_modifier = temperature_modifier

    self._clustering_algorithm = None if clustering_distance == 0 else DBSCAN(eps=clustering_distance)
    self._cumulative_enlarge_ratio = 0.0
    self._clustering_times = 0
    self._clustering_summary_path = output_directory.joinpath("clustering-summary")
    self._neighbor_radius = None if neighbor_radius == 0 else neighbor_radius
    self._clustering_evaluation_path = output_directory.joinpath("clustering-evaluation")
    self._clustering_evaluation_path.mkdir(exist_ok=True)
    self._clustering_nmis = []

    self._saved_gaussian_parameters: tuple | None = None

    # prepare for visualize
    if visualize_views != 0:
      if cameras is None or output_directory is None:
        raise TypeError
      self._visualizer = ADCVisualizer(cameras=cameras, views=visualize_views, seed="ADC visualizer")
      self._render_configuration = Namespace(
        debug=False,
        antialiasing=True,
        compute_cov3D_python=False,
        convert_SHs_python=False,
      )
      self._distribution_columns = 256
      self._distribution_block_size = 11
      self._distribution_limit_mark = 2
      self._visualization_font = ImageFont.load_default()

      self._distribution_output_directory = output_directory.joinpath("attribute-distributions")
      rmtree(path=self._distribution_output_directory, ignore_errors=True)
      self._distribution_output_directory.mkdir(exist_ok=False)

      self._adc_visualize_directory = output_directory.joinpath("foresee-adc-visualize")
      rmtree(path=self._adc_visualize_directory, ignore_errors=True)
      self._adc_visualize_directory.mkdir(exist_ok=False)
    else:
      self._visualizer = None

  def _make_single_distribution(
    self,
    exploring: torch.Tensor,
    decision: torch.Tensor,
    name: str,
  ) -> Image.Image:
    # clamp the value first
    exploring = clamp_extreme_values(data=exploring, minimum=0.001, maximum=0.001)
    decision = clamp_extreme_values(data=decision, minimum=0.001, maximum=0.001)
    # find out the range of data so that we can align their axis
    minimum = min(exploring.min(), decision.min())
    maximum = max(exploring.max(), decision.max())
    # make the range
    if name in ("volume",):
      ranges = torch.logspace(
        torch.log(minimum + 1e-8),
        torch.log(maximum),
        self._distribution_columns + 1,
        base=torch.e,
        device=exploring.device,
      )
      ranges[0] = minimum
    else:
      ranges = torch.linspace(minimum, maximum, self._distribution_columns + 1, device=exploring.device)

    # render the bar about gaussians selected during exploring
    exploring_bar = VisualizeDistribution.visualize(
      data=VisualizeDistribution.DataSpecification(
        data=exploring,
        ranges=ranges,
      ),
      size=VisualizeDistribution.SizeSpecification(
        block=self._distribution_block_size,
        limit_mark=self._distribution_limit_mark,
      ),
      color=VisualizeDistribution.ColorSpecification(
        background=torch.zeros(size=(3,), device=exploring.device),
        maximum_value=VisualizeDistribution.CommonMaximumColors[0],
        limit_mark=VisualizeDistribution.CommonLimitColor,
      ),
      device=exploring.device,
    )
    # render for gaussians selected by final decision
    decision_bar = VisualizeDistribution.visualize(
      data=VisualizeDistribution.DataSpecification(
        data=decision,
        ranges=ranges,
      ),
      size=VisualizeDistribution.SizeSpecification(
        block=self._distribution_block_size,
        limit_mark=self._distribution_limit_mark,
      ),
      color=VisualizeDistribution.ColorSpecification(
        background=torch.zeros(size=(3,), device=exploring.device),
        maximum_value=VisualizeDistribution.CommonMaximumColors[1],
        limit_mark=VisualizeDistribution.CommonLimitColor,
      ),
      device=exploring.device,
    )

    # convert to image
    image_array = torch.cat((exploring_bar, decision_bar), dim=1)
    image = Image.fromarray(
      (image_array.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(numpy.uint8),
    )

    # add name and return
    return add_title(
      image=image,
      text=name,
      height=self._distribution_block_size * 2,
      font=self._visualization_font,
      color=("black", "white"),
    )

  def _visualize_decision(
    self,
    gaussians: TrainingGaussian,
    decision: torch.Tensor,
    difference: torch.Tensor,
    selector: torch.Tensor,
  ) -> None:
    if self._visualizer is None:
      return

    # we visualize metric difference on saved gaussians and the distribution of metrics and parameters of
    #  gaussians selected during exploration and in final decision

    ## visualization on saved gaussian goes first
    # rendering_gaussians = MaskedGaussianModelView(gaussian=gaussians, mask=selector)
    # for index, metric_name in enumerate(self._metrics_to_use):
    #   visualization = visualize_value_on_gaussian(
    #     gaussian=rendering_gaussians,
    #     cameras=(adc_view.view for adc_view in self._visualizer.views),
    #     value=ValueSpecification(
    #       value=difference[:, index],
    #       zero_level=0,
    #       clamp=(0.001, 0.001),
    #       apply_cutting_point=True,
    #     ),
    #     colors=ValueVisualizationColorPack(
    #       zero_level_color=CommonZeroLevelColor,
    #       positive_color=CommonGoodColor,
    #       negative_color=CommonBadColor,
    #     ),
    #   )
    #   for adc_view, image in zip(self._visualizer.views, visualization.images, strict=True):
    #     adc_view.register_visualization(
    #       annotation=(
    #         f"difference of {metric_name}",
    #         f"L={visualization.clamped_minimum}",
    #         f"RL={visualization.raw_minimum}",
    #         f"H={visualization.clamped_maximum}",
    #         f"RH={visualization.raw_maximum}",
    #       ),
    #       image=image,
    #     )
    ## we can now save the visualization
    self._visualizer.save(
      destination=self._adc_visualize_directory.joinpath(f"{self._training_progress.iteration:05d}.png"),
    )

    ## then goes the distributions
    # metrics = self._capture_gaussian_metrics(gaussian=gaussians)
    # distributions = tile_images(
    #   images=tuple(
    #     map(
    #       partial(
    #         ImageOps.pad,
    #         size=(
    #           self._distribution_block_size * self._distribution_columns,
    #           self._distribution_block_size * 5,
    #         ),
    #         color="black",
    #         centering=(1, 1),
    #       ),
    #       (
    #         self._make_single_distribution(
    #           exploring=values[selector],
    #           decision=values[decision],
    #           name=name,
    #         )
    #         for name, values in chain(
    #           (
    #             (
    #               "volume",
    #               gaussians.get_scaling[:, 0]
    #               * gaussians.get_scaling[:, 1]
    #               * gaussians.get_scaling[:, 2]
    #               * torch.pi
    #               * 4
    #               / 3,
    #             ),
    #             ("opacity", gaussians.get_opacity[..., 0]),
    #           ),
    #           (
    #             (
    #               metric_name,
    #               metrics[:, index],
    #             )
    #             for index, metric_name in enumerate(self._metrics_to_use)
    #           ),
    #         )
    #       ),
    #     ),
    #   ),
    #   columns=1,
    # )
    # add_title(
    #   image=distributions,
    #   text=f"{self._training_progress.iteration:05d}",
    #   height=self._distribution_block_size * 3,
    #   font=self._visualization_font,
    #   color=("black", "white"),
    # ).save(self._distribution_output_directory.joinpath(f"{self._training_progress.iteration:05d}.png"))

  class _ForeseerReturnType(NamedTuple):
    # the final decision as a selector: this is a bit mask over gaussians before exploration took place
    decision: torch.Tensor
    # calculated and reduced difference of metrics
    difference: torch.Tensor
    # bit mask describing gaussian corresponding to each line of difference
    #  not all gaussians densified during exploration have valid difference value since some of which may
    #  be pruned. This selector selects gaussians with valid difference values. This selector forms a
    #  subset of all gaussians selected during exploration (self._saved_selector), and decision forms a
    #  subset of this selector
    selector: torch.Tensor

  def _foreseer(self, gaussians: TrainingGaussian) -> _ForeseerReturnType:
    """Evaluate exploring result and decide good and bad densifies.

    After the gaussian model is densified with exploring densify classifiers and trained for several steps,
     this method evaluate the drop of metrics during these steps and split densified gaussians into two
     groups: good gaussians that should be densified and bad gaussians that were densified during exploration
     but shown no benefit to the metrics.
    """
    # capture current metric: metric before next densify step after the point we look into the future
    current_metrics = self._capture_gaussian_metrics(gaussian=gaussians)
    # read exported mapping from original points to densified points
    split_mapping = gaussians.registers["split-mapping"]
    clone_mapping = gaussians.registers["clone-mapping"]
    # calculate metric difference: since splitting one gaussian into more than 2 gaussians is supported,
    #  we calculate for clone and split separately so that the dimension will never mismatch
    split_result = calculate_metric_difference(
      source_points=self._saved_selection,
      source_metrics=self._saved_metrics,
      mapping=split_mapping,
      current_points=gaussians.point_id,
      current_metrics=current_metrics,
    )
    clone_result = calculate_metric_difference(
      source_points=self._saved_selection,
      source_metrics=self._saved_metrics,
      mapping=clone_mapping,
      current_points=gaussians.point_id,
      current_metrics=current_metrics,
    )
    # build selector selecting effective (densified gaussian with valid data) out of saved gaussians
    selector = torch.cat(
      (
        torch.where(self._saved_selector)[0][split_result.effective_source_points],
        torch.where(self._saved_selector)[0][clone_result.effective_source_points],
      ),
    )
    # reshape the difference into a single tensor
    # since difference shows the result of subtracting current metric value from recorded value
    #  a positive value indicates the metric value has decreased
    difference = torch.cat(
      (
        split_result.absolute_difference,
        clone_result.absolute_difference,
      ),
      dim=0,
    )
    # reduce the difference into a single scalar for each gaussian
    reduced_difference = self._metric_reducer(difference)

    # pick candidates
    candidate_selector = torch.zeros(
      size=(len(reduced_difference),),
      dtype=bool,
      device=reduced_difference.device,
    )
    if self._temperature_modifier <= 0:
      # deterministic candidate picking
      candidate_selector[reduced_difference >= 0] = True
    else:
      # random candidate picking
      ## calculate probability for each gaussian
      probabilities = 2.0 / (
        2.0
        + torch.exp(
          -self._temperature_modifier
          * (
            1 - self._training_progress.densify_steps_remaining / self._training_progress.total_densify_steps
          )
          * reduced_difference,
        )
      )
      ## generate a random vector
      sampled_value = torch.rand(size=(len(probabilities),), device=probabilities.device)
      ## fill selector
      candidate_selector[sampled_value <= probabilities] = True

    # generate the final selector (final decision)
    densify_selector = torch.zeros_like(self._saved_selector)
    densify_selector[selector[candidate_selector]] = True

    # reorder selector and reduced difference
    _, ordering = torch.sort(selector)
    selector_bit_mask = torch.zeros_like(self._saved_selector)
    selector_bit_mask[selector] = True

    return _ForeseerDensifyClassifier._ForeseerReturnType(
      decision=densify_selector,
      difference=reduced_difference[ordering],
      selector=selector_bit_mask,
    )

  def _visualize_clustering(
    self,
    gaussians: TrainingGaussian,
    grouping: torch.Tensor,
    group_score: torch.Tensor,
  ) -> None:
    if self._visualizer is None:
      return
    temporary_grouping = grouping.clone()
    temporary_grouping[temporary_grouping == -1] = temporary_grouping.max() + 1
    group_colors = torch.rand(size=(temporary_grouping.max() + 1, 3), device=grouping.device)
    group_colors[-1, :] = 0  # make gaussians assigned to no group rendered as black
    decision_color = torch.zeros(size=(temporary_grouping.max() + 1, 3), device=grouping.device)
    temporary_group_score = torch.cat((group_score, torch.as_tensor([0], device=group_score.device)))
    decision_color[temporary_group_score > 0] = torch.as_tensor(
      [0.3137, 0.9255, 0.9373],
      device=grouping.device,
    )
    decision_color[temporary_group_score <= 0] = torch.as_tensor(
      [0.9373, 0.3255, 0.3137],
      device=grouping.device,
    )
    decision_color[-1, :] = 0  # make gaussians assigned to no group rendered as black

    for _adc_view in self._visualizer.views:
      _view, _original_name = _adc_view.view, _adc_view.name
      _render_result = render(
        viewpoint_camera=_view,
        pc=gaussians,
        pipe=self._render_configuration,
        bg_color=torch.zeros(size=(3,), device="cuda"),
        override_color=group_colors[grouping],
      )["render"]
      _adc_view.register_visualization(
        annotation=("group w/o decision",),
        image=_render_result,
      )

      _render_result = render(
        viewpoint_camera=_view,
        pc=gaussians,
        pipe=self._render_configuration,
        bg_color=torch.zeros(size=(3,), device="cuda"),
        override_color=decision_color[grouping],
      )["render"]
      _adc_view.register_visualization(
        annotation=("group w/ decision",),
        image=_render_result,
      )

  def _gather_neighbor_features(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if self._neighbor_radius is None:
      return torch.empty(size=(len(features), 0), device=features.device)

    _step_size = 100
    collected_features = torch.empty_like(features)
    for i in range(0, len(positions), _step_size):
      distances = torch.linalg.norm(positions[i : i + _step_size, None, :] - positions[None, ...], dim=-1)
      selector = distances <= self._neighbor_radius
      collected_features[i : i + _step_size, ...] = (features[None, ...] * selector[..., None]).sum(
        dim=1
      ) / selector.sum(dim=1, keepdim=True)

    return collected_features

  def _enlarge_selection(self, selection: torch.Tensor, gaussians: TrainingGaussian) -> torch.Tensor:
    # first, collect features for clustering
    features = []

    ## position of gaussians: we will map them linearly into range [0, 1] on each dimension
    raw_position = gaussians.get_xyz
    minimum_coordinates = raw_position.min(dim=0, keepdim=True)[0]
    maximum_coordinates = raw_position.max(dim=0, keepdim=True)[0]
    features.append((raw_position - minimum_coordinates) / (maximum_coordinates - minimum_coordinates))

    ## volume of the gaussians: the distance between minimum and maximum volume can be huge, we map them into
    ##  range [0, 1] logarithmically
    scales = gaussians.get_scaling
    raw_volume = scales[:, 0] * scales[:, 1] * scales[:, 2] * 4 * torch.pi / 3
    log_volume = torch.log(raw_volume)
    minimum_volume = log_volume.min()
    maximum_volume = log_volume.max()
    features.append(((log_volume - minimum_volume) / (maximum_volume - minimum_volume))[:, None])

    ## opacity of the gaussians: this feature falls in range [0, 1] naturally
    features.append(gaussians.get_opacity)

    ## positional grad: we map their norm linearly into range [0, 1]
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads[grads.isnan()] = 0.0
    grad_norm = torch.norm(grads, dim=-1)
    minimum_grad = grad_norm.min()
    maximum_grad = grad_norm.max()
    features.append(((grad_norm - minimum_grad) / (maximum_grad - minimum_grad))[:, None])

    ## all extra metrics collected: we map each of them linearly into range [0, 1]
    for getter in self._metric_getter:
      metric = getter(gaussians)
      minimum_metric = metric.min()
      maximum_metric = metric.max()
      value_component = (metric - minimum_metric) / (maximum_metric - minimum_metric)
      features.append(value_component if len(value_component.shape) == 2 else value_component[:, None])

    ## collect features into a single tensor
    features = torch.cat(features, dim=-1)
    ## merge features from nearby gaussians
    features = torch.cat(
      (
        features,
        self._gather_neighbor_features(
          features=features,
          positions=gaussians.get_xyz,
        ),
      ),
      dim=-1,
    )

    # do clustering over features collected
    clustering_result = self._clustering_algorithm.fit_predict(features, out_dtype="int64")
    # collect (covert) the result back to torch
    clustering_result = torch.as_tensor(clustering_result, device="cuda")

    # enlarge selection to gaussians that are not explored but share similar features with gaussians evaluated
    #  to be densified during exploration

    ## basic result: decisions made by the exploration
    result = selection.clone()

    ## calculate score for each group recognized during clustering
    ##  each explored gaussian that will be densified gives +1 to its group
    ##  each explored gaussian that will not be densified gives -1 to its group
    group_scores = torch.zeros(size=(clustering_result.max() + 1,), device="cuda")
    positive_gaussians = torch.logical_and(self._saved_selector, selection)
    negative_gaussians = torch.logical_and(self._saved_selector, ~selection)
    positive_groups = clustering_result[positive_gaussians]
    negative_groups = clustering_result[negative_gaussians]
    group_scores[positive_groups[positive_groups >= 0]] += 1
    group_scores[negative_groups[negative_groups >= 0]] -= 1

    ## evaluation NMI and conditional entropy
    entropy_result = _compute_nmi(
      clustering_result=clustering_result,
      positive_gaussians=positive_gaussians,
      negative_gaussians=negative_gaussians,
    )
    self._clustering_evaluation_path.joinpath(f"{self._training_progress.iteration:05d}").write_text(
      "\n".join(
        [
          f"entropy = {group_metric['H_Y_given_Cj']:.7f}"
          for group_metric in entropy_result["per_cluster"].values()
        ],
      ),
    )
    self._clustering_nmis.append(entropy_result["NMI"])

    ## visualization
    self._visualize_clustering(
      gaussians=gaussians,
      grouping=clustering_result,
      group_score=group_scores,
    )

    ## select gaussian with group assigned (gaussian that cannot be assigned into a group is marks as -1)
    ##  with positive score to be densified
    grouped_gaussians = clustering_result >= 0
    group_selector = group_scores[clustering_result[grouped_gaussians]] > 0
    densify_gaussians = torch.where(grouped_gaussians)[0][group_selector]
    result[densify_gaussians] = True

    _originally_proposed = selection.sum().item()
    _selection_after_enlarge = result.sum().item()
    self._cumulative_enlarge_ratio += (_selection_after_enlarge - _originally_proposed) / _originally_proposed
    self._clustering_times += 1
    print(f"clustering: {_originally_proposed} -> {_selection_after_enlarge} gaussians as densify candidate")

    return result

  def _visualize_render(self, gaussians: TrainingGaussian) -> None:
    if self._visualizer is None:
      return

    for _adc_view in self._visualizer.views:
      _view, _original_name = _adc_view.view, _adc_view.name
      _render_result = render(
        viewpoint_camera=_view,
        pc=gaussians,
        pipe=self._render_configuration,
        bg_color=torch.zeros(size=(3,), device="cuda"),
      )["render"]
      _l1_loss_image = (
        torch.abs(_render_result - _view.original_image).mean(dim=0, keepdims=True).expand(3, -1, -1)
      )
      _ssim_loss_image = 1 - (
        ssim(_render_result, _view.original_image, with_map=True)[1]
        .mean(dim=0, keepdims=True)
        .expand(3, -1, -1)
      )
      _adc_view.register_visualization(annotation=(_original_name,), image=_render_result)
      _adc_view.register_visualization(annotation=("l1",), image=_l1_loss_image)
      _adc_view.register_visualization(annotation=("ssim",), image=_ssim_loss_image)

  def _visualize_explore(self, gaussians: TrainingGaussian, selector: torch.Tensor, name: str) -> None:
    if self._visualizer is None:
      return

    metrics = self._capture_gaussian_metrics(gaussian=gaussians)
    rendering_gaussians = MaskedGaussianModelView(gaussian=gaussians, mask=selector)
    for index, metric_name in enumerate(self._metrics_to_use):
      visualization = visualize_value_on_gaussian(
        gaussian=rendering_gaussians,
        cameras=(adc_view.view for adc_view in self._visualizer.views),
        value=ValueSpecification(
          value=metrics[:, index][selector],
          zero_level=0,
          clamp=(0.001, 0.001),
          apply_cutting_point=False,
        ),
        colors=ValueVisualizationColorPack(
          zero_level_color=CommonZeroLevelColor,
          positive_color=torch.ones(size=(3,), device="cuda"),
          negative_color=CommonGoodColor,
        ),
      )
      for adc_view, image in zip(self._visualizer.views, visualization.images, strict=True):
        adc_view.register_visualization(
          annotation=(
            f"{metric_name} on {name}",
            f"L={visualization.clamped_minimum}",
            f"RL={visualization.raw_minimum}",
            f"H={visualization.clamped_maximum}",
            f"RH={visualization.raw_maximum}",
          ),
          image=image,
        )

  def _apply_explore_classifiers(self, gaussians: TrainingGaussian) -> torch.Tensor:
    result = None
    for name, classifier in self._exploring_classifiers:
      this_result = classifier(gaussians=gaussians, visualizer=None)
      # visualize the gaussian selected and their current metrics
      # self._visualize_explore(gaussians=gaussians, selector=this_result, name=name)
      result = this_result if result is None else result.logical_or(this_result)
    return result

  def _restrict_densify_number(self, selector: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    # calculate number of gaussians we can densify at this step
    if self._target_gaussian == 0:
      # no limitation, return the selector as is
      return selector
    densify_limit = int(
      (1 - self._training_progress.densify_steps_remaining / self._training_progress.total_densify_steps)
      ** (1 / 3)
      * self._target_gaussian,
    ) - len(selector)

    # check if we have exceeded the limitation
    if selector.sum() <= densify_limit:
      # it is fine, return as is
      return selector

    # sample selector since we exceeded the limitation
    if weight is None:
      # sample uniformly
      sampled_indices = sample(range(selector.sum().item()), k=densify_limit)
    else:
      # sample with respect to the weight
      sampled_indices = torch.multinomial(
        input=weight,
        num_samples=densify_limit,
        replacement=False,
      )
    resampled_selector = selector.clone()
    resampled_selector[:] = False
    resampled_selector[torch.where(selector)[0][sampled_indices]] = True
    return resampled_selector

  @override
  def __call__(
    self,
    gaussians: TrainingGaussian,
    visualizer: ADCVisualizer | None,
  ) -> torch.Tensor:
    if self._saved_gaussian_parameters is None:
      if self._training_progress.iteration == self._training_progress.final_densify_iteration:
        # write a summary about ratio
        self._clustering_summary_path.write_text(
          f"mean enlarge ratio: {self._cumulative_enlarge_ratio / self._clustering_times}",
        )
        maximum_nmis = max(self._clustering_nmis)
        minimum_nmis = min(self._clustering_nmis)
        nmis_distance = maximum_nmis - minimum_nmis
        self._clustering_evaluation_path.joinpath("summary").write_text(
          "\n".join(
            [
              f"mean NMI: {sum(self._clustering_nmis) / len(self._clustering_nmis):.7f}",
              *[
                f"{step:03d}-{'-' * int(108 * (nmi - minimum_nmis) / nmis_distance)}>"
                for step, nmi in enumerate(self._clustering_nmis)
              ],
              "\n\n",
              *[f"{nmi:.7f}" for nmi in self._clustering_nmis],
            ],
          ),
        )
        # we skip the last densify step since we will have no change to evaluate it later
        return torch.zeros(size=(len(gaussians.get_xyz),), dtype=bool, device=gaussians.get_xyz.device)

      # This is the first time we reach this iteration, exploring
      ## do initial visualization: place rendered images
      self._visualize_render(gaussians=gaussians)
      ## save current gaussian parameters so that we can come back later
      self._saved_gaussian_parameters = deepcopy(gaussians.capture())
      ## generate exploring densify selection with underlying classifiers
      self._saved_selector = self._apply_explore_classifiers(gaussians=gaussians)
      ## save the selection and current metrics so that we can evaluate the selection later
      self._saved_selection = gaussians.point_id[self._saved_selector].clone()
      self._saved_metrics = self._capture_gaussian_metrics(gaussian=gaussians)[self._saved_selector].clone()
      ## return the exploring selection
      return self._saved_selector

    # Back from the future, we can now make decision with what we saw in the future
    ## evaluate effect of densify selections and make final decision
    result = self._foreseer(gaussians=gaussians)
    ## restore gaussian parameters
    gaussians.restore(model_args=self._saved_gaussian_parameters, training_args=self._optimization_config)
    if self._clustering_algorithm is not None:
      ## use clustering to discover more gaussians that should be densified
      selection = self._enlarge_selection(selection=result.decision, gaussians=gaussians)
      ## restrict the number of gaussians to densify
      selection = self._restrict_densify_number(selector=selection, weight=None)
    else:
      ## use the result directly
      selection = result.decision
      ## restrict the number of gaussians to densify
      selection = self._restrict_densify_number(
        selector=selection,
        weight=None if self._temperature_modifier > 0 else result.difference[selection],
      )
    ## adjust the training iteration counter back
    self._training_progress.step_to(target=self._training_progress.last_densify_iteration)
    ## visualize the decision: we have to place it here after the gaussian has been restored since we need to
    ##  render on the saved gaussian and use the parameters on it
    self._visualize_decision(
      gaussians=gaussians,
      decision=selection,
      difference=result.difference,
      selector=result.selector,
    )
    ## clear internal state
    self._saved_gaussian_parameters = None
    return selection


Classifier = _ForeseerDensifyClassifier
