from json import load as json_load
from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import numpy

source_path = Path(argv[1])
with source_path.open("r") as data_file:
  data = json_load(data_file)
step = numpy.array([item["step"] for item in data])
l1 = numpy.array([item["l1"] for item in data])
vl1 = numpy.array([item["vl1"] for item in data])
relative_error = numpy.array([item["r-error"] for item in data])
relative_validation_error = numpy.array([item["rv-error"] for item in data])
learning_rate_drops = []
_current_learning_rate = None
for item in data:
  if _current_learning_rate is None:
    _current_learning_rate = item["lr"]
    continue
  if _current_learning_rate != item["lr"]:
    learning_rate_drops.append(item["step"])
    _current_learning_rate = item["lr"]

figure, first_axis = plt.subplots(figsize=(160, 90))
first_axis.plot(step, l1, color="red", label="L1")
first_axis.plot(step, vl1, color="red", label="validation L1")
first_axis.set_ylabel("L1")
first_axis.set_yscale("log")

second_axis = first_axis.twinx()
second_axis.plot(step, relative_error, color="green", label="relative error")
second_axis.plot(step, relative_validation_error, color="blue", label="relative validation error")
second_axis.set_ylabel("relative error")
second_axis.set_yscale("log")

for drop_step in learning_rate_drops:
  first_axis.axvline(x=drop_step, color="orange", linestyle="--", linewidth=2)

first_axis.set_xlabel("iterations")
lines1, labels1 = first_axis.get_legend_handles_labels()
lines2, labels2 = second_axis.get_legend_handles_labels()
first_axis.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(argv[2])
