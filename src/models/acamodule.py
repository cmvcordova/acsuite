from typing import Any, List, Tuple, Union, Literal

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric 
from torchmetrics.classification import AUROC


## Heavily borrows from ashleve/lightning-hydra-template's 
## MNISTLitModule class
## https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py

class ACAModule(LightningModule):
    """ Activity-Cliff Aware representation learning through multiple encoding schemes.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        objective: Literal['binary_classification', 'reconstruction'],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.modules.loss

    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        ## ignore the 'ignore' tag that will be proposed by the logger
        self.save_hyperparameters(logger=False)

        self.net = net
        # loss function
        ### if siamese autoencoder.... elif autoencoder...
        self.criterion = criterion
        self.objective = objective

        if self.objective == 'binary_classification':
            self.train_auroc =  AUROC(task='binary')
            self.val_auroc = AUROC(task='binary')
            self.test_auroc = AUROC(task='binary')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        ## define the default forward pass depending on
        ## associated autoencoder
    def forward(self, x:torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, x) if self.objective == 'reconstruction' else self.criterion(logits, y)
        preds = logits # Only using this since BCEWithLogitsLoss is used, change if loss function changes
        return loss, preds, y
        
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self.model_step(batch) 
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.objective == 'binary_classification':
            self.train_auroc(preds, y) if self.train_auroc else None
            self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        #loss, preds, targets = self.model_step(batch)
        loss, preds, y = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.objective == 'binary_classification':
            self.val_auroc(preds, y) if self.val_auroc else None
            self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if self.objective == 'binary_classification':
            auroc = self.val_auroc.compute()  # get current auroc acc
            self.val_auroc_best(auroc)  # update best so far val acc
            # log `val_auroc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
    
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
    _ = ACAModule(None, None, None, None, None)