from __future__ import annotations

from argparse import ArgumentParser
from itertools import batched, chain
from math import ceil, log10
from pathlib import Path
from statistics import mean
from typing import Any, NamedTuple

import torch


class MetricsRecord(NamedTuple):
  position: tuple[float, float, float]
  scale: tuple[float, float, float]
  rotation: tuple[float, float, float, float]
  opacity: float
  grad: float
  ldl: float
  l1: float
  ssim: float
  frequency: float


class PointRecord(NamedTuple):
  point_id: int
  metrics: MetricsRecord


class DataLine(NamedTuple):
  step: float
  position: tuple[float, float, float]
  scale: tuple[float, float, float]
  rotation: tuple[float, float, float, float]
  opacity: float
  grad: float
  ldl: float
  l1: float
  ssim: float
  frequency: float

  delta_ldl: float
  delta_l1: float
  delta_ssim: float


def _pack_data_line(data: DataLine) -> torch.Tensor:
  return torch.as_tensor(
    (
      data.step,
      *data.position,
      *data.scale,
      *data.rotation,
      data.opacity,
      data.grad,
      data.ldl,
      data.l1,
      data.ssim,
      data.frequency,
      data.delta_ldl,
      data.delta_l1,
      data.delta_ssim,
    ),
  )


class SplitRecord(NamedTuple):
  source: int
  destination: tuple[int, int]


class CloneRecord(NamedTuple):
  source: int
  destination: int


class StepRecord(NamedTuple):
  step: int
  percentage_step: float
  start_points: list[int]
  metrics: list[MetricsRecord]
  clones: list[CloneRecord]
  splits: list[SplitRecord]
  pruned: list[int]


def _handle_shot_trace(trace: dict[str, Any]) -> tuple[list[MetricsRecord], list[int]]:
  metrics = [
    MetricsRecord(
      position=tuple(trace["position"][i].tolist()),
      scale=tuple(trace["scale"][i].tolist()),
      rotation=tuple(trace["rotation"][i].tolist()),
      opacity=trace["opacity"][i].item(),
      grad=trace["metrics"]["grad"][i].item(),
      ldl=trace["metrics"]["ldl"][i].item(),
      l1=trace["metrics"]["l1"][i].item(),
      ssim=trace["metrics"]["ssim"][i].item(),
      frequency=trace["metrics"]["frequency"][i].item(),
    )
    for i in range(len(trace["points"]))
  ]
  start_points = trace["points"].tolist()

  return metrics, start_points


def _pack_step(traces: list[Path]) -> StepRecord:
  data = [torch.load(traces.pop())]
  while True:
    temporary_data = torch.load(traces[-1])
    if temporary_data["step"] != data[0]["step"]:
      break
    traces.pop()
    data.append(temporary_data)

  clones = []
  splits = []
  pruned = []

  for item in data:
    if item["event_type"] == "shot":
      metrics, start_points = _handle_shot_trace(trace=item)
    elif item["event_type"] == "split":
      source_points = len(item["from"])
      destination_points = len(item["to"])
      splits = [
        SplitRecord(source=source, destination=destination)
        for source, destination in zip(
          item["from"].tolist(),
          batched(item["to"].tolist(), n=destination_points // source_points, strict=True),
          strict=True,
        )
      ]
    elif item["event_type"] == "clone":
      clones = [
        CloneRecord(source=source, destination=destination)
        for source, destination in zip(item["from"].tolist(), item["to"].tolist(), strict=True)
      ]
    elif item["event_type"] == "prune":
      pruned = item["points"].tolist()

  return StepRecord(
    step=data[0]["step"],
    percentage_step=data[0]["percentage_step"],
    start_points=start_points,
    metrics=metrics,
    clones=clones,
    splits=splits,
    pruned=pruned,
  )

def _align(value: int, maximum: int) -> str:
  return str(value).rjust(ceil(log10(maximum + 1)))

def _collect_data(
  traces: list[Path],
  *,
  start_samples: int,
  start_filecount: int,
  total_files: int,
) -> torch.Tensor:
  total = len(traces)
  samples = 0

  current_points: dict[int, PointRecord]

  last_step = -1

  results = []

  # load last trace which must be the final shot
  final_trace = torch.load(traces.pop())
  if final_trace["event_type"] != "shot":
    raise ValueError
  metrics, start_points = _handle_shot_trace(trace=final_trace)
  current_points = {
    i: PointRecord(point_id=i, metrics=metric) for i, metric in zip(start_points, metrics, strict=True)
  }
  last_step = final_trace["step"]

  # load update steps
  while len(traces) != 1:
    local_finished = total - len(traces)
    print(
      f"Local: {_align(local_finished, total)} / {total}; "
      f"Global: {_align(local_finished + start_filecount, total_files)} / {total_files}; "
      f"Lines collected: local {samples:10d}, global {samples + start_samples:10d}",
    )
    record = _pack_step(traces)
    # validation
    ## pruned points shall not exist in current points
    if any(pruned_point in current_points for pruned_point in record.pruned):
      raise ValueError
    ## cloned points should be in current points unless it is pruned afterwards
    if any(
      point not in current_points and point not in record.pruned
      for point in chain.from_iterable((clone.source, clone.destination) for clone in record.clones)
    ):
      raise ValueError
    ## split points acts like cloned points but should have themselves removed
    if any(
      point not in current_points and point not in record.pruned
      for split in record.splits
      for point in split.destination
    ) or any(split.source in current_points for split in record.splits):
      raise ValueError
    ## validation done

    cloned = {clone.source: clone.destination for clone in record.clones}
    split = {split.source: split.destination for split in record.splits}

    def _select_children(
      point_id: int,
      current_points: dict[int, PointRecord],
      cloned: dict[int, int],
      split: dict[int, tuple[int, int]],
    ) -> tuple[PointRecord, ...]:
      if point_id in cloned:
        try_list = [point_id, cloned[point_id]]
      elif point_id in split:
        try_list = [*split[point_id]]
      else:
        try_list = []
      existing_points = [point_id for point_id in try_list if point_id in current_points]
      return tuple(current_points[index] for index in existing_points)

    for i, metric in zip(record.start_points, record.metrics, strict=True):
      children = _select_children(point_id=i, current_points=current_points, cloned=cloned, split=split)
      if not children:
        continue
      mean_ldl = mean(child.metrics.ldl for child in children)
      mean_l1 = mean(child.metrics.l1 for child in children)
      mean_ssim = mean(child.metrics.ssim for child in children)

      results.append(
        _pack_data_line(
          data=DataLine(
            step=record.percentage_step,
            position=metric.position,
            scale=metric.scale,
            rotation=metric.rotation,
            opacity=metric.opacity,
            grad=metric.grad,
            ldl=metric.ldl,
            l1=metric.l1,
            ssim=metric.ssim,
            frequency=metric.frequency,
            delta_ldl=(mean_ldl - metric.ldl) / (last_step - record.step),
            delta_l1=(mean_l1 - metric.l1) / (last_step - record.step),
            delta_ssim=(mean_ssim - metric.ssim) / (last_step - record.step),
          ),
        ),
      )
      samples += 1

    current_points = {
      i: PointRecord(
        point_id=i,
        metrics=metric,
      )
      for i, metric in zip(record.start_points, record.metrics, strict=True)
    }
    last_step = record.step

  # simple validation
  first_trace = torch.load(traces.pop())
  if first_trace["event_type"] != "initialize":
    raise ValueError

  return torch.stack(results, dim=0)


parser = ArgumentParser()
parser.add_argument("data", type=Path, nargs="+")
parser.add_argument("--destination", type=Path, required=True)
arguments = parser.parse_args()

file_list = [sorted(data_path.iterdir()) if data_path.is_dir() else data_path for data_path in arguments.data]
total_files = sum(len(files) for files in file_list if isinstance(files, list))
final_results = []
finished_files = 0
collected_samples = 0

for files in file_list:
  if isinstance(files, Path):
    data_slice = torch.load(files)
  else:
    files_count = len(files)
    data_slice = _collect_data(
      traces=files,
      start_samples=collected_samples,
      start_filecount=finished_files,
      total_files=total_files,
    )
    finished_files += files_count

  collected_samples += len(data_slice)
  final_results.append(data_slice)

torch.save(torch.concatenate(final_results, dim=0), arguments.destination)
