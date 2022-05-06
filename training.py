"""This is a generic PyTorch training loop that can be adapted for different problems."""

from itertools import count
from typing import Callable, Union

import numpy as np
import torch
import torch.utils.data.dataloader

import logging


def default_batch_callback(
    epoch: int,
    batch: int,
    output: torch.Tensor,
    target: torch.Tensor,
    loss: float,
) -> None:
    return None


def default_epoch_callback(
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> None:
    return None


def default_loss_improvement_callback(
    epoch: int,
    loss: float,
) -> None:
    return None


def _do_batches(
    epoch: int,
    model: torch.nn.Module,
    data_loader: torch.utils.data.dataloader.DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optim: torch.optim.Optimizer,
    training: bool,
    callback: Callable[[int, int, torch.Tensor, torch.Tensor, float], None],
):
    model.train(training)
    batch_losses = []

    for batch, (predictors, target) in enumerate(data_loader):
        model_output = model(predictors)
        batch_loss = loss_function(model_output, target)
        batch_losses.append(float(batch_loss))

        if training:
            optim.zero_grad()
            batch_loss.backward()
            optim.step()

        callback(epoch, batch, model_output, target, float(batch_loss))

    epoch_loss = float(np.mean(batch_losses))
    return epoch_loss


def train(
    model: torch.nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optim: torch.optim.Optimizer,
    train_loader: torch.utils.data.dataloader.DataLoader,
    validation_loader: torch.utils.data.dataloader.DataLoader,
    max_epochs: Union[int, None] = None,
    early_termination: Union[int, None] = None,
    train_batch_callback: Callable[
        [int, int, torch.Tensor, torch.Tensor, float], None
    ] = default_batch_callback,
    validation_batch_callback: Callable[
        [int, int, torch.Tensor, torch.Tensor, float], None
    ] = default_batch_callback,
    loss_improvement_callback: Callable[
        [int, float], None
    ] = default_loss_improvement_callback,
    epoch_callback: Callable[[int, float, float], None] = default_epoch_callback,
):
    # Initialize state for early termination monitoring
    best_validation_loss = float("inf")
    best_epoch = -1

    # This is the main epoch loop
    if max_epochs is None and early_termination is None:
        raise ValueError(
            f"At least one of max_epochs ({max_epochs}) or early_termination ({early_termination}) must be specified"
        )

    if max_epochs is not None:
        epoch_iterator = range(max_epochs)
    else:
        epoch_iterator = count()

    for epoch in epoch_iterator:

        epoch_train_loss = _do_batches(
            epoch,
            model,
            train_loader,
            loss_function,
            optim,
            training=True,
            callback=train_batch_callback,
        )

        # Evalute the loss on the test set
        # Don't compute gradients
        with torch.no_grad():
            epoch_validation_loss = _do_batches(
                epoch,
                model,
                validation_loader,
                loss_function,
                optim,
                training=False,
                callback=validation_batch_callback,
            )

        epoch_callback(epoch, float(epoch_train_loss), float(epoch_validation_loss))

        logging.info(f"    Epoch {epoch}: loss (train): {epoch_train_loss:.4f}")

        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_epoch = epoch
            flag = "**"

            loss_improvement_callback(epoch, epoch_validation_loss)
        else:
            flag = "  "

        logging.info(
            f" {flag} Epoch {epoch}: loss (test): {epoch_validation_loss:.4f}  best epoch: {best_epoch}  best loss:{best_validation_loss:.4f} {flag}"
        )
        if early_termination is not None and epoch >= best_epoch + early_termination:
            logging.info(
                f"No improvement in {early_termination} epochs.  Terminating early."
            )
            break  # Terminate early

    return best_validation_loss, best_epoch
