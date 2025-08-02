"""Classifiers used in densification."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from inspect import getfile
from pathlib import Path
from typing import Literal, Protocol

from scene.densification_classifiers_typing import GaussianADCClassifier


class ClassifierReady(Protocol):
  def register_densify_classifier(self, name: str, classifier: GaussianADCClassifier) -> None: ...
  def register_prune_classifier(self, name: str, classifier: GaussianADCClassifier) -> None: ...


def _attach_classifier_direct(
  target: ClassifierReady,
  mode: Literal["densify", "prune"],
  classifier: GaussianADCClassifier,
  classifier_name: str | None = None,
) -> None:
  if classifier_name is None:
    classifier_name = Path(getfile(classifier.__class__)).stem

  (target.register_densify_classifier if mode == "densify" else target.register_prune_classifier)(
    name=classifier_name,
    classifier=classifier,
  )


def _find_classifier(name: str) -> type[GaussianADCClassifier]:
  classifier_path = Path(__file__).parent.joinpath("densify_classifiers", f"{name}.py")
  if not classifier_path.is_file():
    message = f"Cannot find classifier {name}: no such file [{classifier_path}]"
    raise FileNotFoundError(message)
  spec = spec_from_file_location("", classifier_path)
  module = module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.Classifier


def attach_classifier(
  target: ClassifierReady,
  mode: Literal["densify", "prune"],
  classifier: GaussianADCClassifier | type[GaussianADCClassifier] | str,
  classifier_name: str | None = None,
  *args,  # noqa: ANN002
  **kwargs,  # noqa: ANN003
) -> None:
  if isinstance(classifier, GaussianADCClassifier):
    return _attach_classifier_direct(
      target=target,
      mode=mode,
      classifier=classifier,
      classifier_name=classifier_name,
    )
  if isinstance(classifier, type) and issubclass(classifier, GaussianADCClassifier):
    return _attach_classifier_direct(
      target=target,
      mode=mode,
      classifier=classifier(*args, **kwargs),
      classifier_name=classifier_name,
    )
  if isinstance(classifier, str):
    return attach_classifier(
      *args,
      target=target,
      mode=mode,
      classifier=_find_classifier(name=classifier),
      classifier_name=classifier_name or classifier,
      **kwargs,
    )
  message = f"failed to attach classifier {classifier}: "
  "it neither an instance or subclass of GaussianADCClassifier or a string"
  raise TypeError(message)
