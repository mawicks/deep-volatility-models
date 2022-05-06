"""This is a generic PyTorch training loop that can be adapted for different problems."""

from typing import Callable

import numpy as np
import torch

import logging


def default_batch_callback(
    epoch: int, batch: int, output: torch.Tensor, target: torch.Tensor, loss: float
) -> None:
    return None


def default_epoch_callback(
    epoch: int, train_loss: float, validation_loss: float
) -> None:
    return None


def default_model_improvement_callback(epoch: int, validation_loss: float) -> None:
    return None


def do_batches(
    epoch: int,
    model: torch.nn.Module,
    data_loader: torch.utils.data.dataloader.DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], float],
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
    model,
    loss_function,
    optim,
    train_loader,
    validation_loader,
    max_epochs=None,
    early_termination=None,
    train_batch_callback=default_batch_callback,
    validation_batch_callback=default_batch_callback,
    model_improvement_callback=default_model_improvement_callback,
    epoch_callback=default_epoch_callback,
):
    # Initialize state for early termination monitoring
    best_validation_loss = float("inf")
    best_epoch = -1

    # This is the main epoch loop
    for epoch in range(max_epochs):

        train_epoch_loss = do_batches(
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
            validation_epoch_loss = do_batches(
                epoch,
                model,
                validation_loader,
                loss_function,
                optim,
                training=False,
                callback=validation_batch_callback,
            )

            epoch_callback(epoch, train_epoch_loss, validation_epoch_loss)

        logging.info(f"    Epoch {epoch}: loss (train): {train_epoch_loss:.4f}")

        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss
            best_epoch = epoch
            flag = "**"

            model_improvement_callback(epoch, validation_epoch_loss)
        else:
            flag = "  "
        logging.info(
            f" {flag} Epoch {epoch}: loss (test): {validation_epoch_loss:.4f}  best epoch: {best_epoch}  best loss:{best_validation_loss:.4f} {flag}"
        )
        if early_termination is not None and epoch >= best_epoch + early_termination:
            logging.info(
                f"No improvement in {early_termination} epochs.  Terminating early."
            )
            break  # Terminate early

    return best_validation_loss, best_epoch
