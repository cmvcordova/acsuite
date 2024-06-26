from typing import Any, Literal, Optional, Dict, Tuple
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from torchmetrics.classification import AUROC, Accuracy
from torchmetrics.regression import MeanSquaredError
from src.models.components.losses import RMSELoss
class ACAPPModule(LightningModule):

    """ Module for training a property predictor on top of a 
    pretrained encoder """
    
    def __init__(
        self,
        net: torch.nn.Module,
        task: Literal["classification", "regression"],
        optimizer: torch.optim.Optimizer,
        num_classes: int = 1,
        criterion: Optional[torch.nn.Module] = None,
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
        #self.compile = compile # add later
        assert self.task in ["classification", "regression"], f"Unexpected task: {self.task}"
        
        if self.criterion is None:
            if self.task == "classification":
                self.criterion = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
            elif self.task == "regression":
                self.criterion = RMSELoss()

        if self.task == "classification":
            if num_classes == 1:
                self.metric_name = 'AUROC'
                self.train_metric = AUROC(num_classes=1, task="binary")
                self.val_metric = AUROC(num_classes=1, task="binary")
                self.test_metric = AUROC(num_classes=1, task="binary")
            ## todo: add multiclass auroc later
            else:
                self.metric_name = 'Accuracy'
                self.train_metric = Accuracy(num_classes=num_classes)
                self.val_metric = Accuracy(num_classes=num_classes)
                self.test_metric = Accuracy(num_classes=num_classes)

        elif self.task == "regression":
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

        if task == "classification":
            self.val_metric_best = MaxMetric()
        elif task == "regression":
            self.val_metric_best = MinMetric()

        ## define the default forward pass depending on associated model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

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
        x, y = batch
        preds = self.forward(x)
        if self.task == "regression":
            preds = preds.squeeze(-1)
        if self.task == "classification" and self.num_classes == 1:
            y = y.unsqueeze(-1)
        loss = self.criterion(preds, y.float())
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_metric(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"train/{self.metric_name}", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
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

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_metric(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test/{self.metric_name}", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
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

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
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
    _ = ACAPPModule(None, None, None, None, None)