from json import loads
from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import numpy

iterations = []
dldl = []
dl1 = []
dssim = []

source_path = Path(argv[1])
with source_path.open("r") as data_file:
  _current_iteration = 700
  for data_line in data_file:
    data = loads(data_line)
    dldl.append(data["dldl%"])
    dl1.append(data["dl1%"])
    dssim.append(data["dssim%"])
    iterations.append(_current_iteration)
    _current_iteration += 100

iterations = numpy.array(iterations)
dldl = numpy.log(numpy.array(dldl) / 100)
dl1 = numpy.log(numpy.array(dl1) / 100)
dssim = numpy.log(numpy.array(dssim) / 100)

y_label = "log-difference(x)"

plt.figure(figsize=(16, 9))
plt.plot(iterations, dldl, label="ldl")
plt.plot(iterations, dl1, label="l1")
plt.plot(iterations, dssim, label="SSIM")
plt.xlabel("iterations")
plt.ylabel(y_label)
plt.legend()
plt.grid(True)
plt.savefig(argv[2])
