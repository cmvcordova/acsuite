from typing import Any, Literal, Optional, List
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import AUROC, Accuracy
from torchmetrics.regression import MeanSquaredError
from src.models.components.loss.loss import RMSELoss


class ACAPPModule(LightningModule):

    """ Module for training a property predictor on top of a 
    pretrained encoder"""
    
    def __init__(
        self,
        net: torch.nn.Module,
        task: Literal['classification', 'regression'],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: torch.nn.Module
        
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        ## ignore the 'ignore' tag that will be proposed by the logger
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = criterion
                
        # metric objects for calculating and averaging accuracy across batches
        if task == "classification":
            if isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
                self.metric_name = 'AUROC'
                self.train_metric = AUROC(task='binary')
                self.val_metric = AUROC(task='binary')
                self.test_metric = AUROC(task='binary')

            ## todo: add support in options for accuracy
            #self.train_metric = Accuracy(task='binary')
            #self.val_metric = Accuracy(task='binary')
            #self.test_metric = Accuracy(task='binary')

        elif task == "regression":
            if isinstance(criterion, RMSELoss):
                self.metric_name = 'rmse'
                self.train_metric = MeanSquaredError(squared=False)
                self.val_metric = MeanSquaredError(squared=False)
                self.test_metric = MeanSquaredError(squared=False)

            elif isinstance(criterion, torch.nn.modules.loss.MSELoss):
                self.metric_name = 'mse'
                self.train_metric = MeanSquaredError(squared=True)
                self.val_metric = MeanSquaredError(squared=True)
                self.test_metric = MeanSquaredError(squared=True)

            else:
                raise ValueError(f"Unsuported loss metric {criterion} for the regression task")
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_metric_best = MaxMetric()

        ## define the default forward pass depending on associated model
    def forward(self, x:torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        ## classification tasks:
        self.val_metric.reset()
        self.val_metric_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        y = y.unsqueeze(-1)
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        #preds = torch.argmax(preds, dim=1) #classification, removed assuming
        #the classification problem is binary and the criterion is BCEWithLogitsLoss
        return loss , preds, y
        
    def training_step(self, batch: Any, batch_idx: int):
        ## Required
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
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_metric(preds, targets)
        #self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        print('validation step')
        self.log(f"val/{self.metric_name}", self.val_metric, 
                 on_step=True, 
                 on_epoch=False, 
                 prog_bar=True)

    #Use when incorporating classification tasks
    def on_validation_epoch_end(self):
        metric = self.val_metric.compute()  # get current val acc
        self.val_metric_best(metric)  # update best so far val acc
        # log `val_metric_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        print('on_validation_epoch_end')
        self.log(f"val/{self.metric_name}_best", self.val_metric_best.compute(), 
                 on_step=False,
                 on_epoch=True, 
                 sync_dist=True, 
                 prog_bar=True
            )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_metric(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test/{self.metric_name}", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        ## Required
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
    
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
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
    _ = ACAPPModule(None, None, None, None, None)