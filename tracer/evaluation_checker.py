from json import load
from pathlib import Path
from sys import argv

with Path(argv[1]).open("r") as f:
  data = load(f)

current = {item[0]: item[1] for item in data["current"]}
mapping = {item[0]: item[1] for item in data["mapping"]}

v = tuple([] for _ in range(len(data["last"][0][1])))
for pid, values in data["last"]:
  if pid not in mapping:
    continue
  if not mapping[pid]:
    continue
  _values = tuple([] for _ in range(len(v)))
  for cpid in mapping[pid]:
    for i in range(len(_values)):
      _values[i].append(current[cpid][i])
  for i in range(len(_values)):
    v[i].append((values[i] - sum(_values[i]) / len(_values[i])) ** 2)
v = tuple(sum(item) / len(item) for item in v)
print(v)
