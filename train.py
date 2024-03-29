#!/usr/bin/env python

import sys

# needed because dgllife can't find rdkit otherwise...
from rdkit import Chem

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (32768, rlimit[1]))

import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from routines.ai_routine import AIRoutine
from common.cfg_utils import get_config, get_run_config

def train(cfg):

    callbacks = []
    if cfg.project is not None:
        if cfg.resume_id:
            run = wandb.init(project=cfg.project, id=cfg.resume_id, resume=True)
            cfg = get_run_config(run, cfg)
            artifact = run.use_artifact(f"{cfg.project}/model-{run.id}:latest", type='model')
            artifact_dir = artifact.download()
            checkpoint_file = artifact_dir + "/model.ckpt"
            routine = AIRoutine.from_checkpoint(cfg, checkpoint_file)
        
        logger = WandbLogger(project=cfg.project, name=cfg.name, log_model="all")
        logger.log_hyperparams(cfg)
        if "SLURM_JOB_ID" in os.environ:
            logger.log_hyperparams({ "slurm_job_id": os.environ["SLURM_JOB_ID"] })

        val_metric = {
            "classification": "val_auroc",
            "regression": "val_r2"
        }[cfg.task]
        checkpoint_callback = ModelCheckpoint(monitor=val_metric, mode="max", save_last=True, every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        routine = AIRoutine(cfg)
    else:
        routine = AIRoutine(cfg)
        logger = None
        
    routine.fit(logger, callbacks)

if __name__ == "__main__":
    if len(sys.argv) > 1 and '=' not in sys.argv[1]:
        cfg_name = sys.argv[1]
    else:
        cfg_name = "default"
    cfg = get_config(cfg_name=cfg_name)
    train(cfg)
