
from typing import Union, Callable

import paddle
from paddle import nn
import tqdm
from utils.logging_utils import AverageMeter


class TransformationTrainer:
    """Class to train and evaluate transformation models."""
    def __init__(self,
                 old_model: Union[nn.Layer, paddle.jit.TranslatedLayer],
                 new_model: Union[nn.Layer, paddle.jit.TranslatedLayer],
                 side_info_model: Union[nn.Layer, paddle.jit.TranslatedLayer],
                 **kwargs) -> None:
        """Construct a TransformationTrainer module.
        Args:
            old_model: A model that returns old embedding given x.
            new_model: A model that returns new embedding given x.
            side_info_model: A model that returns side-info given x.
        """

        self.old_model = old_model
        self.old_model.eval()
        self.new_model = new_model
        self.new_model.eval()
        self.side_info_model = side_info_model
        self.side_info_model.eval()

    def train(self,
              train_loader: paddle.io.DataLoader,
              model: nn.Layer,
              criterion: Callable,
              optimizer: paddle.optimizer.Optimizer,
              switch_mode_to_eval: bool) -> float:
        """Run one epoch of training.
        Args:
            train_loader: Data loader to train the model.
            model: Model to be trained.
            criterion: Loss criterion module.
            optimizer: A torch optimizer object.
            device: Device the model is on.
        Returns:
            Average loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")

        if switch_mode_to_eval:
            model.eval()
        else:
            model.train()

        for i, (images, _) in enumerate(train_loader):
            with paddle.no_grad():
                old_feature = self.old_model(images)
                new_feature = self.new_model(images)
                side_info = self.side_info_model(images)

            recycled_feature = model(old_feature, side_info)
            loss = criterion(new_feature, recycled_feature)

            losses.update(loss.item(), images.shape[0])

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        return losses.avg

    def validate(self,
                 val_loader: paddle.io.DataLoader,
                 model: nn.Layer,
                 criterion: Callable) -> float:
        """Run validation
        Args:
            val_loader: Data loader to evaluate the model.
            model: Model to be evaluated.
            criterion: Loss criterion module.
        Returns:
            Average loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        model.eval()

        for i, (images, _) in enumerate(val_loader):
            with paddle.no_grad():
                old_feature = self.old_model(images)
                new_feature = self.new_model(images)
                side_info = self.side_info_model(images)

                recycled_feature = model(old_feature, side_info)
                loss = criterion(new_feature, recycled_feature)

            losses.update(loss.item(), images.shape[0])

        return losses.avg
