# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pathlib
import time
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import neptune
import cv2
import click
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import yacs
import filetype
from torch import nn
from torch.nn import TripletMarginLoss
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
from yacs.config import CfgNode

import spai.data.data_finetune
from spai.config import get_config
from spai.models import build_cls_model
from spai.data import build_loader, build_loader_test
from spai.lr_scheduler import build_scheduler
from spai.models.sid import AttentionMask
from spai.onnx import compare_pytorch_onnx_models
from spai.optimizer import build_optimizer
from spai.logger import create_logger
from spai.utils import (
    load_pretrained,
    save_checkpoint,
    get_grad_norm,
    find_pretrained_checkpoints,
    inf_nan_to_num
)
from spai.models import losses
from spai import metrics
from spai import data_utils


def _cuda_enabled() -> bool:
    # Allow forcing CPU mode and avoid probing CUDA on incompatible drivers.
    if os.environ.get("SPAI_FORCE_CPU", "0") == "1":
        return False
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

cv2.setNumThreads(1)
logger: Optional[logging.Logger] = None


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--batch-size", type=int,
              help="Batch size for a single GPU.")
@click.option("--learning-rate", type=float)
@click.option("--data-path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="path to dataset")
@click.option("--csv-root-dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--lmdb", "lmdb_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an LMDB file storage that contains the files defined in the "
                   "dataset's CSV file. If this option is not provided, the data will be "
                   "loaded from the filesystem.")
@click.option("--pretrained",
              type=click.Path(exists=True, dir_okay=False),
              help="path to pre-trained model")
@click.option("--resume", is_flag=True,
              help="resume from checkpoint")
@click.option("--accumulation-steps", type=int, default=1,
              help="Gradient accumulation steps.")
@click.option("--use-checkpoint", is_flag=True,
              help="Whether to use gradient checkpointing to save memory.")
@click.option("--amp-opt-level", type=click.Choice(["O0", "O1", "O2"]), default="O1",
              help="mixed precision opt level, if O0, no amp is used")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              help="root of output folder, the full path is "
                   "<output>/<model_name>/<tag> (default: output)")
@click.option("--tag", type=str,
              help="tag of experiment")
@click.option("--local_rank", type=int, default=0,
              help="local_rank for distributed training")
@click.option("--test-csv", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a CSV with test data. If this option is provided after the "
                   "validation of each epoch, a testing will also take place. This option "
                   "intends to facilitate understanding the progression of the generalization "
                   "ability of a model among the epochs and should not be used for selecting "
                   "the final model. This option can be repeated several times. For each provided "
                   "csv file, a separate testing run is going to take place.")
@click.option("--test-csv-root-dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Root directory for the relative paths included into the test csv files. "
                   "If this option is omitted, the parent directory of each test csv file will "
                   "be used as the root dir for the paths it contains. If this option is provided "
                   "a single time, it will be used as the root dir for all the test csv files. If "
                   "it is provided multiple times, each value will be matched with a corresponding "
                   "test csv file. In that case, the number of provided test csv files and the "
                   "number of provided root directories should match. The order of the provided "
                   "arguments will be used for the matching.")
@click.option("--data-workers", type=int,
              help="Number of worker processes to be used for data loading.")
@click.option("--disable-pin-memory", is_flag=True)
@click.option("--data-prefetch-factor", type=int)
@click.option("--save-all", is_flag=True)
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def train(
    cfg: Path,
    batch_size: Optional[int],
    learning_rate: Optional[float],
    data_path: Path,
    csv_root_dir: Optional[Path],
    lmdb_path: Optional[Path],
    pretrained: Optional[Path],
    resume: bool,
    accumulation_steps: int,
    use_checkpoint: bool,
    amp_opt_level: str,
    output: Path,
    tag: str,
    local_rank: int,
    test_csv: list[Path],
    test_csv_root_dir: list[Path],
    data_workers: Optional[int],
    disable_pin_memory: bool,
    data_prefetch_factor: Optional[int],
    save_all: bool,
    extra_options: tuple[str, str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = data_path.parent
    config = get_config({
        "cfg": str(cfg),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "data_path": str(data_path),
        "csv_root_dir": str(csv_root_dir),
        "lmdb_path": str(lmdb_path),
        "pretrained": str(pretrained) if pretrained is not None else None,
        "resume": resume,
        "accumulation_steps": accumulation_steps,
        "use_checkpoint": use_checkpoint,
        "amp_opt_level": amp_opt_level,
        "output": str(output),
        "tag": tag,
        "local_rank": local_rank,
        "test_csv": [str(p) for p in test_csv],
        "test_csv_root": [str(p) for p in test_csv_root_dir],
        "data_workers": data_workers,
        "disable_pin_memory": disable_pin_memory,
        "data_prefetch_factor": data_prefetch_factor,
        "opts": extra_options
    })
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    # Set a fixed seed to all the random number generators.
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    if config.TRAIN.SCALE_LR:
        # Linear scale the learning rate according to total batch size - may not be optimal.
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZ