import wandb
import numpy as np
from typing import Any, List, Tuple, Dict, Optional, Callable

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader

from src.trainers.base import Trainer
from src.utils import save

import logging
log = logging.getLogger(__name__)


class BasicTrainer(Trainer):

    def run(self):
        config = self.config.trainer
        opt = optim.Adam(
            self.model.parameters(), 
            lr=config.lr, betas=config.betas, eps=config.eps
        )
        examples_seen = 0
        for epoch in range(config.num_epochs):
            for i, batch in enumerate(self.train_dl):
                examples_seen += self.train_dl.batch_size
                loss, metric = self.model.step(batch) 
                loss.backward()
                opt.step()
                opt.zero_grad()
                wandb.log(
                    {"train/loss": loss.item(), "train/metric": metric}, 
                    step=examples_seen
                )
                if i%config.eval_interval==0:
                    eval_loss, eval_metric = self.test(self.eval_dl)
                    wandb.log(
                        {"eval/loss": eval_loss, "eval/metric": eval_metric}, 
                        step=examples_seen
                    )
            log.info(f"Trained for {epoch+1} epochs")
            log.info(f"Evaluation\tLoss:{eval_loss}\tMetric:{eval_metric}")
        test_loss, test_metric = self.test(self.test_dl)
        wandb.log(
            {"test/loss": test_loss, "test/metric": test_metric}, 
            step=examples_seen
        )
        log.info(f"Test\tLoss:{test_loss}\tMetric:{test_metric}")
        path = self.ckpt_dir/f"model-{examples_seen}-{test_loss:.3f}.pt"
        save(self.model, opt, test_loss, examples_seen, path)
        log.info(f"Saved trained model checkpoint at {path}")
    
    def test(self, dataloader: DataLoader) -> Tuple[np.float64, np.float64]:
        losses, metrics = [], []
        for batch in dataloader:
            with torch.no_grad():
                loss, metric = self.model.step(batch) 
            losses.append(loss.item())
            metrics.append(metric)
        test_loss = np.mean(losses)
        test_metric = np.mean(metrics)
        return test_loss, test_metric