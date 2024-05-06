from typing import Any, Literal, Optional, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from torchmetrics.classification import AUROC, Accuracy
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from src.models.components.metrics import CollapseLevel
from src.models.components.losses import (
    BarlowTwinsLoss, 
    RMSELoss, 
    NegativeCosineSimilarityLoss,
    SiamACLoss)

class ACAModule(LightningModule):
    """ 
    Activity-Cliff Aware representation learning through multiple encoding schemes.
    """
    def __init__(
        self,
        net: torch.nn.Module,
        task: Literal['classification', 'regression', 'reconstruction','self_supervision'],
        optimizer: torch.optim.Optimizer,
        num_classes: Optional[int] = 1,
        criterion: Optional[torch.nn.Module] = None,
        input_type: Literal['single', 'pair'] = 'single',
        compile: Optional[bool] = False,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        ## ignore the 'ignore' tag that will be proposed by the logger
        print(net)
        self.save_hyperparameters(logger=False)
        self.net = net
        self.task = task
        self.num_classes = num_classes
        self.criterion = criterion or self.default_criterion()
        self.input_type = input_type
        
        if task != "self_supervision":
            self.initialize_metrics(task, num_classes)
        else:
            if self.criterion is None:
                raise ValueError("Specific criterion must be provided for self-supervised learning.")
            self.train_metric = self.val_metric = self.test_metric = None
            self.metric_name = None
                        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if x2 is not None:
            return self.net(x1, x2)
        return self.net(x1).squeeze(-1)

    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        if getattr(self, 'val_loss', None):
            self.val_loss.reset()
        if getattr(self, 'val_metric', None):
            self.val_metric.reset()
        if getattr(self, 'val_metric_best', None):
            self.val_metric_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        if self.input_type == "pair":
            x1, x2, _ = batch
            x1 = x1.squeeze(1)
            x2 = x2.squeeze(1)
            
            if isinstance(self.criterion, SiamACLoss):
                out_1 = self.forward(x1)
                out_2 = self.forward(x2)
                loss = self.criterion(x1, out_1, x2, out_2)
            if isinstance(self.criterion, NegativeCosineSimilarityLoss):
                if self.task == "self_supervision": # simsiam
                    p1, z2 = self.forward(x1, x2)
                    p2, z1 = self.forward(x2, x1)
                    loss = 0.5 * (self.criterion(p2, z1) + self.criterion(p1, z2))
                    ## minimum possible value should be -1
                elif self.task == "reconstruction": # siamese autoencoder
                    recon_x1, recon_x2 = self.forward(x1, x2)
                    loss = self.criterion(recon_x1, recon_x2)
            elif isinstance(self.criterion, BarlowTwinsLoss):
                ## Check, hasn't been extensively tested
                loss = self.criterion(z1, z2)
            else:
                raise ValueError("Unsupported criterion for self-supervised learning.")  
            return loss, None, None
        else:
            x, y = batch
            x = x.squeeze(1)
            logits = self.forward(x)
            if logits.dim() > y.dim():
                y = y.unsqueeze(1)
            y = y.float()
            loss = self.criterion(logits, y)   
            return loss, logits, y
        
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        if preds is not None:
            self.train_metric(preds, targets)
            self.log(f"train/{self.metric_name}", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        #loss, preds, targets = self.model_step(batch)
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        if preds is not None:
            self.val_metric(preds, targets)
            self.log(f"val/{self.metric_name}", self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        #return loss
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if getattr(self, 'val_metric', None):
            metric = self.val_metric.compute()  # get current val metric
            if self.val_metric_best:
                self.val_metric_best(metric)  # update best so far val metric
            # log `val_metric_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            if self.metric_name:
                self.log(f"val/{self.metric_name}_best", self.val_metric_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        if preds is not None:
            self.test_metric(preds, targets)
            self.log(f"test/{self.metric_name}", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        #return loss
    
    def on_test_epoch_end(self):
        pass
    
    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, _ = batch  
        logits = self.forward(x)
        if self.task == "classification":
            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
            else: ## multiclass
                probs = torch.softmax(logits, dim=1)
            return probs
        return logits  # For regression tasks, return raw outputs

    def configure_optimizers(self):
        ## Required
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
    
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def initialize_metrics(self, task: str, num_classes: int, out_dim: int = 2048):
        """Dynamically initializes metrics based on the task type."""
        if task == "reconstruction":
            self.val_metric_best = MinMetric()
            self.metric_name = 'MAE'
            metric = MeanAbsoluteError()
            self.train_metric = self.val_metric = self.test_metric = metric

        elif task == "classification":
            self.val_metric_best = MaxMetric()
            if num_classes == 1:
                self.metric_name = 'AUROC'
                metric = AUROC(num_classes=1, task="binary")
                self.train_metric = self.val_metric = self.test_metric = AUROC(num_classes=1, task="binary")
            else:
                self.metric_name = 'Accuracy'
                metric = Accuracy(num_classes=num_classes)
                self.train_metric = self.val_metric = self.test_metric = Accuracy(num_classes=num_classes)

        elif task == "regression":
            self.val_metric_best = MinMetric()
            if isinstance(self.criterion, RMSELoss):
                self.metric_name = 'RMSE'
                metric = MeanSquaredError(squared=True)
                self.train_metric = self.val_metric = self.test_metric = MeanSquaredError(squared=True)
            else:
                self.metric_name = 'MSE'
                metric = MeanSquaredError(squared=False)
                self.train_metric = self.val_metric = self.test_metric = MeanSquaredError(squared=False)
        elif task == "self_supervision":
            self.val_metric_best = MinMetric()
            self.collapse_metric = CollapseLevel(out_dim=out_dim, w=0.9)
        else:
            self.metric = None  # unsupported loss
            
    def default_criterion(self):
        """Define default criterion based on the task."""
        if self.task == "classification":
            return nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        elif self.task == "regression":
            return RMSELoss()
        elif self.task == "reconstruction":
            return nn.BCEWithLogitsLoss()
        raise ValueError("No default criterion for the given task.")


if __name__ == "__main__":
    _ = ACAModule(None, None, None, None, None)