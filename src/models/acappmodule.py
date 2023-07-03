
## write a class that takes a pretrained AE
## module, keeps the encoder minus the first layer
## and appends a property prediction MLP to the end,
## on any of the MoleculeACE tasks
## you'd probably write a second class that takes
## the module and unrolls it inside a single lightning module.

## REMEMBER TO CHANGE NET TO AUTOENCODER IN THE ACAMODULE.PY FILE
## FOR CONSISTENCY


class ACAPPModule(LightningModule):

    """ Module for training a property predictor on top of a 
    pretrained autoencoder"""
    
    def __init__(
        self,
        property_predictor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.modules.loss,
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
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        """
        Use when incorporating classification tasks, currently unused
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        """

        ## define the default forward pass depending on
        ## associated autoencoder
    def forward(self, x:torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        ## classification tasks:
        #self.val_acc.reset()
        #self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, _ = batch
        x_logits = self.forward(x)
        loss = self.criterion(x_logits, x)
        #preds = torch.argmax(logits, dim=1) classification
        return loss #, preds, y
        
    def training_step(self, batch: Any, batch_idx: int):
        ## Required
        #loss, preds, targets = self.model_step(batch)
        loss = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        #self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        #loss, preds, targets = self.model_step(batch)
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        #self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    """
    Use when incorporating classification tasks
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
    """
    def test_step(self, batch: Any, batch_idx: int):
        #loss, preds, targets = self.model_step(batch)
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        #self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
    
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
    _ = ACAPPModule(None, None, None, None)