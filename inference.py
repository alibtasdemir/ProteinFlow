import time

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import GPUtil
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from models.proteinflow_wrapper import ProteinFlowModule
from utils.experiments import get_pylogger, LengthDataset

"""
# Configuration for inference on SE(3) diffusion experiments.
defaults:
  - base
  - _self_

inference:

  # Use this to write with date-time stamp.
  name: run_${now:%Y-%m-%d}_${now:%H-%M}
  seed: 123
  ckpt_path: weights/published.ckpt
  output_dir: inference_outputs/

  use_gpu: True
  num_gpus: 2

  interpolant:
    min_t: 1e-2
    rots:
      corrupt: True
      sample_schedule: exp
      exp_rate: 10
    trans:
      corrupt: True
      sample_schedule: linear
    sampling:
      num_timesteps: 100
    self_condition: True

  samples:

    # Number of backbone samples per sequence length.
    samples_per_length: 10

    # Minimum sequence length to sample.
    min_length: 60

    # Maximum sequence length to sample.
    max_length: 128

    # gap between lengths to sample. i.e. this script will sample all lengths
    # in range(min_length, max_length, length_step)
    length_step: 1

    # Subset of lengths to sample. If null, sample all targets.
    length_subset: null

    overwrite: False
"""

log = get_pylogger(__name__)


class Sampler:

    def __init__(self, cfg: DictConfig):
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Setup directories
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(self._infer_cfg.output_idr, self._ckpt_name, self._infer_cfg.name)
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Read checkpoint
        self._flow_module = ProteinFlowModule.load_from_checkpoint(checkpoint_path=ckpt_path)
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit=8
        )[:self._infer_cfg.num_gpus]
        log.info(f'Using devices: {devices}')
        eval_dataset = LengthDataset(self._samples_cfg)
        dataloader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        trainer = Trainer(
            accelarator="gpu",
            strategy="ddp",
            devices=devices
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')


if __name__ == "__main__":
    run()
