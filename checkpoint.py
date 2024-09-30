"""Define Checkpoint class.
"""
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

class Checkpoint:
    """A Checkpoint class to hold the state of a trained model, regardless of
    being trained on CPU or GPU.

    Examples:
        Load checkpoint::
            from deepbio import Checkpoint

            checkpoint = Checkpoint(model, optimizer)
            checkpoint.load("checkpoint.pth.tar")
            best_score = checkpoint.best_score
            start_epoch = checkpoint.start_epoch

        Save checkpoint::my_dict
            from deepbio import Checkpoint

            checkpoint = Checkpoint(model, optimizer, epoch, best_score)
            checkpoint.save("checkpoint.pth.tar")

    See also:
        https://github.com/e-lab/pytorch-toolbox/blob/master/convert-save-load.
        md

    Args:
        model (object): The model object.
        optimizer (object, optional): The optimizer object.
        epoch (int, optional): The trained epoch.
        best_score (float, optional): The score of the best-scoring epoch.
    """
    def __init__(self, model, optimizer=None, epoch=0, best_score=0):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_score = best_score

        self._logger = logging.getLogger("deepbio.Checkpoint")

    def __repr__(self):
        return "Checkpoint({!r}, {!r}, {!r}, {!r})".format(self.model,
                                                           self.optimizer,
                                                           self.epoch,
                                                           self.best_score)

    def load(self, path):
        """Load checkpoint from a file.
        
        Args:
            path (str): The path of the checkpoint file.

        Example:
            checkpoint = Checkpoint(model, optimizer)
            checkpoint.load(checkpoint_file)
            best_score = checkpoint.best_score
            start_epoch = checkpoint.epoch + 1
        """
        self._logger.info("Loading checkpoint: " + path)

        checkpoint = torch.load(path)
        '''print('WARNING: hardcoding here')
        keys_to_remove = [key for key in checkpoint["model_state"].keys() if key.startswith('vit') or key.startswith('head')]
        for key in keys_to_remove:
            del checkpoint["model_state"][key]
        '''
        result = self.model.load_state_dict(checkpoint["model_state"], strict=False)
        print(result)
        self.epoch = checkpoint["epoch"]
        self.best_score = checkpoint["best_score"]

        if self.optimizer:
            self.optimizer = self.optimizer.load_state_dict(
                checkpoint["optimizer_state"])

        self._logger.info(
            "Loaded checkpoint: {path} (epoch {epoch})".format(
                path=path, epoch=self.epoch))

    def save(self, path):
        """Save the checkpoint into a file.

        If the model is an instance of torch.nn.DataParallel, alter the keys of
        the model's state dictionary to support non-DataParallel usage.

        Args:
            path (str): The path to save the checkpoint file.

        Example:
            checkpoint = Checkpoint(model, optimizer, epoch,
                                    max(score, best_score))
            checkpoint.save("checkpoint.pth.tar")
        """
        state_dict = self.model.state_dict()

        # If DataParallel, strip `module.` from keys.
        if isinstance(self.model, nn.DataParallel):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

        torch.save({"model_state": state_dict,
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_score": self.best_score}, path)