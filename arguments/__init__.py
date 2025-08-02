#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from __future__ import annotations

import ast
import builtins
import inspect
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from io import StringIO
from pathlib import Path
from token import COMMENT
from tokenize import TokenInfo, generate_tokens
from typing import TYPE_CHECKING, Any, NamedTuple, override

if TYPE_CHECKING:
  from collections.abc import Set as AbstractSet


class _OptionSpecification(NamedTuple):
  name: str
  value_type: type
  help_message: str | None
  register_shorthand: bool
  accept_multiple_values: bool


class _CommentInformation(NamedTuple):
  help_messages: dict[int, str]
  shorthands: AbstractSet[int]
  multiple_values: AbstractSet[int]


class GroupParameters(ABC):
  @staticmethod
  def _resolve_type(name: str) -> type:
    def _resolve_helper() -> Any:  # noqa: ANN401
      _locals = locals()
      if name in _locals:
        return _locals[name]
      _globals = globals()
      if name in _globals:
        return _globals[name]
      if hasattr(builtins, name):
        return getattr(builtins, name)
      return None

    result = _resolve_helper()
    if not isinstance(result, type):
      raise TypeError
    return result

  @staticmethod
  def _collect_comments(source: StringIO, option_lines: AbstractSet[int]) -> _CommentInformation:
    tokens = generate_tokens(source.readline)
    comment_tokens = [token for token in tokens if token.type == COMMENT]
    comment_tokens.append(
      TokenInfo(
        type=COMMENT,
        string="",
        start=(-1, 0),
        end=(-1, 0),
        line="",
      ),
    )
    messages, shorthands, multiple_values = {}, set(), set()

    end_lineno = None
    string_buffer = ""
    for comment_token in comment_tokens:
      plain_string = comment_token.string[1:].lstrip()
      if end_lineno != comment_token.start[0] - 1 or comment_token.start[0] in option_lines:
        if end_lineno is not None:
          messages[end_lineno] = string_buffer
          string_buffer = ""
          end_lineno = None

        extras = plain_string.split(",")
        if "shorthand" in extras:
          shorthands.add(comment_token.start[0])
        if "multiple_values" in extras:
          multiple_values.add(comment_token.start[0])
        if comment_token.start[0] in option_lines:
          continue

      end_lineno = comment_token.start[0]
      if len(string_buffer) != 0:
        string_buffer += "\n"
      string_buffer += plain_string

    return _CommentInformation(
      help_messages=messages,
      shorthands=shorthands,
      multiple_values=multiple_values,
    )

  @classmethod
  def install_parser(
    cls,
    parser: ArgumentParser,
    *,
    name: str | None = None,
    fill_none: bool = False,
  ) -> None:
    source = inspect.getsource(cls)
    syntax_tree = ast.parse(source, filename="<string>")
    assignments = tuple(
      node for node in syntax_tree.body[0].body if isinstance(node, ast.AnnAssign) and node.simple == 1
    )
    comment_information = GroupParameters._collect_comments(
      source=StringIO(source),
      option_lines={node.lineno for node in assignments},
    )
    options = [
      _OptionSpecification(
        name=node.target.id,
        value_type=GroupParameters._resolve_type(node.annotation.id),
        help_message=comment_information.help_messages.get(node.lineno - 1, None),
        register_shorthand=node.lineno in comment_information.shorthands,
        accept_multiple_values=node.lineno in comment_information.multiple_values,
      )
      for node in assignments
    ]
    cls._options = options
    group = parser.add_argument_group(name or cls.__name__)
    for option in options:
      key = option.name
      value = None if fill_none else getattr(cls, key)

      # allow both --some_option_name and --some-option-name
      keys = [f"--{key}", f"--{key.replace('_', '-')}"]
      # add a shorthand for this option if required
      if option.register_shorthand:
        keys.append(f"-{key[:1]}")

      # extra configurations
      configurations = {
        "default": value,
        "help": option.help_message,
      }
      if option.value_type is bool:
        configurations["action"] = "store_true"
      else:
        configurations["type"] = option.value_type

      if option.accept_multiple_values:
        configurations["nargs"] = "+"

      group.add_argument(*keys, **configurations)

  @classmethod
  def from_parsed(cls, parameters: Namespace) -> GroupParameters:
    structured_parameters = cls()
    parameters = vars(parameters)
    for option in cls._options:
      if option.name in parameters and parameters[option.name] is not None:
        setattr(structured_parameters, option.name, parameters[option.name])
    structured_parameters._postprocess()
    return structured_parameters

  @abstractmethod
  def _postprocess(self) -> None: ...


class ModelParameters(GroupParameters):
  # maximum degree of SH to use
  sh_degree: int = 3

  # root path to scene dataset
  source_path: Path = Path()  # shorthand
  # path to store the trained model
  model_path: Path = Path()  # shorthand

  # name of directory containing scene images in the scene dataset.
  #  Tune this in case the dataset use some unusual names
  images: str = "images"  # shorthand
  depths: str = ""  # shorthand
  resolution: int = -1  # shorthand
  white_background: bool = False  # shorthand
  train_test_exp: bool = False
  data_device: str = "cuda"
  evaluate: bool = False

  @override
  def _postprocess(self) -> None:
    self.source_path = Path(self.source_path).resolve()


class PipelineParameters(GroupParameters):
  # convert SH to RGB color with Python code
  convert_SHs_python: bool = False
  # compute covariance matrix with Python code
  compute_cov3D_python: bool = False

  # debug rendering pipeline
  debug: bool = False

  # apply anti-aliasing during rendering
  antialiasing: bool = False

  # number of views to use in each training step (batch size)
  mv: int = 1

  @override
  def _postprocess(self) -> None: ...


class OptimizationParameters(GroupParameters):
  iterations: int = 30_000
  position_lr_init: float = 0.00016
  position_lr_final: float = 0.0000016
  position_lr_delay_mult: float = 0.01
  position_lr_max_steps: int = 30_000
  feature_lr: float = 0.0025
  opacity_lr: float = 0.025
  scaling_lr: float = 0.005
  rotation_lr: float = 0.001
  exposure_lr_init: float = 0.01
  exposure_lr_final: float = 0.001
  exposure_lr_delay_steps: int = 0
  exposure_lr_delay_mult: float = 0.0
  percent_dense: float = 0.01
  lambda_dssim: float = 0.2
  densification_interval: int = 100
  opacity_reset_interval: int = 3000
  densify_from_iter: int = 500
  densify_until_iter: int = 15_000
  densify_grad_threshold: float = 0.0002
  densify_metric_threshold: float = 6
  depth_l1_weight_init: float = 1.0
  depth_l1_weight_final: float = 0.01
  random_background: bool = False
  optimizer_type: str = "default"

  @override
  def _postprocess(self) -> None: ...


class TrainParameters(GroupParameters):
  # IP address to bind to for WebUI
  ip: str = "127.0.0.1"
  # port to listen on for WebUI
  port: int = 0

  # start debug from certain iteration
  debug_from: int = -1

  # configure torch to detect NaNs and Infs
  detect_anomaly: bool = False

  # iteration to evaluate on test set (novel view)
  test_iterations: int = [7000, 30000]  # multiple_values
  # iteration to save the trained model
  save_iterations: int = [7000, 30000]  # multiple_values

  quiet: bool = False
  seed: int = 0

  # disable the viewer
  disable_viewer: bool = False

  checkpoint_iterations: int = []  # multiple_values
  start_checkpoint: Path = None

  @override
  def _postprocess(self) -> None: ...


def get_combined_args(parser: ArgumentParser) -> Namespace:
  arguments_from_commandline = parser.parse_args()
  config_file = Path(arguments_from_commandline.model_path) / "cfg_args"
  if not config_file.is_file():
    print(f"No config file found at {config_file}")
    return arguments_from_commandline
  saved_config = config_file.read_text()
  saved_arguments = parser.parse_args(saved_config.split("\x00"))
  combined_arguments = vars(arguments_from_commandline).copy()
  for key, value in vars(saved_arguments).items():
    if value is not None:
      combined_arguments[key] = value
  return Namespace(**combined_arguments)
