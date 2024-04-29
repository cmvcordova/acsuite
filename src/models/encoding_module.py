from typing import Any, Literal, Optional, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from torchmetrics.classification import AUROC, Accuracy
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from src.models.components.losses import BarlowTwinsLoss, RMSELoss

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
        
        self.save_hyperparameters(logger=False)
        self.net = net
        self.task = task
        self.num_classes = num_classes
        self.criterion = criterion
        self.input_type = input_type
        assert self.task in ["classification", "regression", "reconstruction"], f"Unexpected task: {self.task}"

        self.train_metric = MeanMetric()
        self.val_metric = MeanMetric()
        self.test_metric = MeanMetric()
        
        if self.criterion is None:
            if self.task == "classification":
                self.criterion = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
            elif self.task == "regression":
                self.criterion = RMSELoss()
            elif self.task == "reconstruction":
                self.criterion = torch.nn.BCEWithLogitsLoss()

        if self.task == "reconstruction":
            self.val_metric_best = MinMetric()
            self.metric_name = 'MAE'
            self.train_metric = MeanAbsoluteError()
            self.val_metric = MeanAbsoluteError()
            self.test_metric = MeanAbsoluteError()
                                
        if self.task == "classification":
            self.val_metric_best = MaxMetric()
            if num_classes == 1:
                self.metric_name = 'AUROC'
                self.train_metric = AUROC(num_classes=1, task="binary")
                self.val_metric = AUROC(num_classes=1, task="binary")
                self.test_metric = AUROC(num_classes=1, task="binary")
            else:
                self.metric_name = 'Accuracy'
                self.train_metric = Accuracy(num_classes=num_classes)
                self.val_metric = Accuracy(num_classes=num_classes)
                self.test_metric = Accuracy(num_classes=num_classes)

        elif self.task == "regression":
            self.val_metric_best = MinMetric()
            if isinstance(self.criterion, RMSELoss):
                self.metric_name = 'RMSE'
                self.train_metric = MeanSquaredError(squared=True)
                self.val_metric = MeanSquaredError(squared=True)
                self.test_metric = MeanSquaredError(squared=True)

            elif isinstance(self.criterion, torch.nn.modules.loss.MSELoss):
                self.metric_name = 'MSE'
                self.train_metric = MeanSquaredError(squared=False)
                self.val_metric = MeanSquaredError(squared=False)
                self.test_metric = MeanSquaredError(squared=False)
            
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if self.input_type == "pair":
            return self.net(x[0], x[1])
        else:
            logits = self.net(x)
        
        logits = logits.squeeze(-1)
        return logits
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric.reset()
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
            x1, x2, y = batch
            output = self.forward((x1, x2))
        else:
            x, y = batch
            output = self.forward(x)
        
        if isinstance(output, tuple): 
            logits, z1, z2 = output  
            loss = self.criterion(z1, z2) 
        else:
            logits = output
            if logits.dim() > y.dim():
                y = y.unsqueeze(1)
            y = y.float()
            loss = self.criterion(logits, y)            
            return loss, logits, y
        
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch) 
        # update and log metrics
        self.train_loss(loss)
        self.train_metric(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"train/{self.metric_name}", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)        
        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        #loss, preds, targets = self.model_step(batch)
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_metric(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/{self.metric_name}", self.val_metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric = self.val_metric.compute()  # get current val metric
        self.val_metric_best(metric)  # update best so far val metric
        # log `val_metric_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(f"val/{self.metric_name}_best", self.val_metric_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_metric(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test/{self.metric_name}", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
    
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
    
if __name__ == "__main__":
    _ = ACAModule(None, None, None, None, None)