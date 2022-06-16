# Standard Python
import datetime as dt
import logging

import os
from typing import Callable, Dict, Iterable, Iterator, Union, Tuple

# Common packages
import click
import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torch.utils.data.dataloader

# Local imports
from deep_volatility_models import data_sources
from deep_volatility_models import stock_data
from deep_volatility_models import mixture_model_stats
from deep_volatility_models import loss_functions
from deep_volatility_models import time_series_datasets
from deep_volatility_models import model_wrappers
from deep_volatility_models import architecture
from deep_volatility_models import training

logging.basicConfig(level=logging.INFO, force=True)

TRAIN_FRACTION = 0.80
DEFAULT_SEED = 24  # Previously 42

EPOCHS = 1000  # 30000
EARLY_TERMINATION = 100  # Was 1000

USE_MIXTURE = True
RISK_NEUTRAL = True
if RISK_NEUTRAL:
    OPT_LEARNING_RATE = 0.000535
    OPT_DROPOUT = 0.001675
    OPT_FEATURE_DIMENSION = 41
    OPT_MIXTURE_COMPONENTS = 4
    OPT_WINDOW_SIZE = 256
    OPT_EMBEDDING_DIMENSION = 4
    OPT_MINIBATCH_SIZE = 124
    OPT_GAUSSIAN_NOISE = 0.002789
    OPT_WEIGHT_DECAY = 1.407138e-06
    USE_BATCH_NORM = False  # risk neutral version has trouble with batch normalization
else:
    # Current values were optimized with hyperopt.  Values shown in comment were used before optimization.
    OPT_LEARNING_RATE = 0.000689  # Previously 0.000375
    OPT_DROPOUT = 0.130894  # Previously 0.50
    OPT_FEATURE_DIMENSION = 86  # Previously 40
    OPT_MIXTURE_COMPONENTS = 3  # Previously 4
    OPT_WINDOW_SIZE = 256  # Previously 64
    OPT_EMBEDDING_DIMENSION = 3  # Previously 10
    OPT_MINIBATCH_SIZE = 248  # Previously 75
    OPT_GAUSSIAN_NOISE = 0.000226  # Previously 0.0025
    OPT_WEIGHT_DECAY = 8.489603e-07  # Previously 5e-9
    # Value of USE_BATCH_NORM wasn't optimized with hyperopt but was set to True.
    USE_BATCH_NORM = True

BETA1 = 0.95
BETA2 = 0.999
ADAM_EPSILON = 1e-8
ACTIVATION = torch.nn.ReLU()
MAX_GRADIENT_NORM = 1.0

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def create_new_model(
    embedding_size=None,
    window_size=OPT_WINDOW_SIZE,
    mixture_components=OPT_MIXTURE_COMPONENTS,
    feature_dimension=OPT_FEATURE_DIMENSION,
    embedding_dimension=OPT_EMBEDDING_DIMENSION,
    gaussian_noise=OPT_GAUSSIAN_NOISE,
    use_batch_norm=USE_BATCH_NORM,
    dropout=OPT_DROPOUT,
    risk_neutral=RISK_NEUTRAL,
    use_mixture=USE_MIXTURE,
):
    if use_mixture:
        default_network_class = architecture.MixtureModel
    else:
        default_network_class = architecture.UnivariateModel

    network = default_network_class(
        window_size,
        1,
        feature_dimension=feature_dimension,
        mixture_components=mixture_components,
        exogenous_dimension=embedding_dimension,
        gaussian_noise=gaussian_noise,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=ACTIVATION,
        risk_neutral=risk_neutral,
    )
    embedding = torch.nn.Embedding(embedding_size, embedding_dimension)

    return network, embedding


def load_existing_model(existing_model, symbols):
    """
    This function loads an existing model and adjusts its embedding
    and encoding objects to accomodate any new symbols in `symbols`

    Arguments:
        existing_model: path - path to existing model
        symbols: List[str] - list of symbols to be trained.

    Returns:
        model_network: torch.Module
        embeddings: torch.Embedding
        encoding: Dict[str, i] - encoding

    Note the list of symbols is required so that the embedding can be extended
    (with values to be trained) to accomodate the new symbol list.

    """

    model = torch.load(existing_model)
    # Dump the old wrapper and keep only the network and the embeddings
    # We'll create a new wrapper
    model_network = model.network
    embeddings = model.embedding
    encoding = model.encoding

    # If there are new symbols since the previous model was trained,
    # extend the encoding and initialize the new embeddings with the
    # mean of the old embedding.  This initialization seems to work
    # better than a random initialization with using a pre-trained
    # model

    new_symbols = set(symbols).difference(set(embeddings.keys()))

    if len(new_symbols) > 0:
        # Extend the encoding for any symbols unknown to the pre-loaded model
        for s in new_symbols:
            encoding[s] = len(encoding)

        # Extend the embedding for any symbols unknown to the pre-loaded model
        embedding_parameters = next(embeddings.parameters())
        mean_embedding = embedding_parameters.mean(dim=0)
        # Extract and use old embedding dimension
        old_embedding_dimension = embedding_parameters.shape[1]

        new_embeddings = (
            mean_embedding.unsqueeze(0)
            .expand(len(new_symbols), old_embedding_dimension)
            .clone()
        )

        # Extend the mean to current number of symbols
        embedding_parameters.data = torch.concat(
            (embedding_parameters, new_embeddings), dim=0
        )

    return model_network, embeddings, encoding


def prepare_data(
    history_loader: Callable[
        [Union[str, Iterable[str]]], Iterator[Tuple[str, pd.DataFrame]]
    ],
    symbol_list: Iterable[str],
    encoding: Dict[str, int],
    window_size: int,
    minibatch_size: int = OPT_MINIBATCH_SIZE,
):
    # Refresh historical data
    logging.info("Reading historical data")
    splits_by_symbol = {}

    # For the purposes of hyperparameter optimization, make sure that
    # changing the window size doesn't change the number of rows.  In
    # other words, we always consume the first 256 points of history,
    # even if we don't use them as context so that first target return
    # in the dataset is always the same, independent of the window
    # size.  Also, this won't work if window_size exceeds 256, so we
    # trap that case:
    if window_size > 256:
        raise ValueError(
            f"Window size of {window_size} isn't allowed.  Window size must be 256 or less"
        )

    skip = 256 - window_size

    for s in symbol_list:
        logging.info(f"Reading {s}")
        i = encoding[s]

        # history_loader can load many symbols at once for multivariate
        # but here we just load the single symbol of interest.  Since we expect
        # just one dataframe, grab it with next() instead of using a combiner()
        # (see stock-data.py).)
        symbol_history = next(history_loader(s))[1]

        # Symbol history is a combined history for all symbols.  We process it
        # one symbols at a time, so get the log returns for the current symbol
        # of interest.
        # log_returns = symbol_history.loc[:, (s, "log_return")]  # type: ignore
        windowed_returns = time_series_datasets.RollingWindow(
            symbol_history.log_return[skip:],
            1 + window_size,
            create_channel_dim=True,
        )
        logging.debug(f"{s} windowed_returns[0]: {windowed_returns[0]}")
        symbol_dataset = time_series_datasets.ContextWindowAndTarget(
            windowed_returns, 1
        )
        symbol_dataset_with_encoding = (
            time_series_datasets.ContextWindowEncodingAndTarget(
                i, symbol_dataset, device=device
            )
        )

        train_size = int(TRAIN_FRACTION * len(symbol_dataset_with_encoding))
        lengths = [train_size, len(symbol_dataset_with_encoding) - train_size]
        train, test = torch.utils.data.random_split(
            symbol_dataset_with_encoding, lengths
        )
        splits_by_symbol[s] = {"train": train, "test": test}

    train_dataset = torch.utils.data.ConcatDataset(
        [splits_by_symbol[s]["train"] for s in symbol_list]
    )
    validation_dataset = torch.utils.data.ConcatDataset(
        [splits_by_symbol[s]["test"] for s in symbol_list]
    )

    train_dataloader = torch.utils.data.dataloader.DataLoader(
        train_dataset, batch_size=minibatch_size, drop_last=True, shuffle=True
    )

    validation_dataloader = torch.utils.data.dataloader.DataLoader(
        validation_dataset,
        batch_size=len(validation_dataset),
        drop_last=True,
        shuffle=True,
    )

    return train_dataloader, validation_dataloader


def make_mixture_loss_function():
    def loss_function(output, target):
        log_p, mu, inv_sigma = output[:3]

        loss = -torch.mean(
            mixture_model_stats.multivariate_log_likelihood(
                target.squeeze(2), log_p, mu, inv_sigma
            )
        )

        if np.isnan(float(loss)):
            logging.error("log_p: ", log_p)
            logging.error("mu: ", mu)
            logging.error("inv_sigma: ", inv_sigma)

        return loss

    return loss_function


def make_loss_function():
    def loss_function(output, target):
        mu, inv_sigma = output[:2]

        loss = -torch.mean(
            loss_functions.univariate_log_likelihood(target.squeeze(2), mu, inv_sigma)
        )

        if np.isnan(float(loss)):
            logging.error("mu: ", mu)
            logging.error("inv_sigma: ", inv_sigma)

        return loss

    return loss_function


def log_mixture_mean_error(epoch, output, target):
    log_p, mu = output[:2]
    mb_size, components, channels = mu.shape
    combined_mu = torch.sum(
        mu * torch.exp(log_p).unsqueeze(2).expand((mb_size, components, channels)),
        dim=1,
    )
    mean_error = torch.mean(target.squeeze(2) - combined_mu, dim=0)
    logging.debug(f"epoch: {epoch} mean_error: {float(mean_error):.5f}")


def make_mixture_validation_batch_logger():
    def log_epoch(epoch, batch, output, target, loss):
        log_p, mu, inv_sigma = output[:3]
        logging.debug(f"last epoch p:\n{torch.exp(log_p)[:6].detach().cpu().numpy()}")
        logging.debug(f"last epoch mu:\n{mu[:6].detach().cpu().numpy()}")
        logging.debug(f"last epoch sigma:\n{inv_sigma[:6].detach().cpu().numpy()}")

        log_mixture_mean_error(epoch, output, target)

    return log_epoch


def log_mean_error(epoch, output, target):
    mu = output[0]
    mean_error = torch.mean(target.squeeze(2) - mu, dim=0)
    logging.debug(f"epoch: {epoch} mean_error: {float(mean_error):.5f}")


def make_validation_batch_logger():
    def log_epoch(epoch, batch, output, target, loss):
        mu, inv_sigma = output[:2]
        logging.debug(f"last epoch mu:\n{mu[:6].detach().cpu().numpy()}")
        logging.debug(f"last epoch sigma:\n{inv_sigma[:6].detach().cpu().numpy()}")

        log_mean_error(epoch, output, target)

    return log_epoch


def make_save_model(model_file, only_embeddings, model, encoding, symbols):
    def save_model(epoch, epoch_loss, prefix=""):
        wrapped_model = model_wrappers.StockModel(
            symbols=symbols,
            encoding=encoding,
            network=model,
            epochs=epoch,
            date=dt.datetime.now(),
            loss=epoch_loss,
        )

        torch.save(wrapped_model, f"{model_file}")

    return save_model


def make_loss_improvement_callback(
    model_file, only_embeddings, model, encoding, symbols
):
    save_model = make_save_model(model_file, only_embeddings, model, encoding, symbols)

    def model_improvement_callback(epoch, epoch_loss):
        save_model(epoch, epoch_loss)

    return model_improvement_callback


def make_epoch_callback(model):
    def epoch_callback(epoch, train_loss, validation_loss):
        logging.debug(f"parameters: {(list(model.embedding.parameters()))}")

    return epoch_callback


def run(
    use_hsmd,
    model_file,
    existing_model,
    symbols,
    refresh,
    risk_neutral,
    only_embeddings,
    use_mixture=USE_MIXTURE,
    max_epochs=EPOCHS,
    early_termination=EARLY_TERMINATION,
    window_size=OPT_WINDOW_SIZE,
    mixture_components=OPT_MIXTURE_COMPONENTS,
    feature_dimension=OPT_FEATURE_DIMENSION,
    embedding_dimension=OPT_EMBEDDING_DIMENSION,
    gaussian_noise=OPT_GAUSSIAN_NOISE,
    minibatch_size=OPT_MINIBATCH_SIZE,
    dropout=OPT_DROPOUT,
    learning_rate=OPT_LEARNING_RATE,
    weight_decay=OPT_WEIGHT_DECAY,
    use_batch_norm=USE_BATCH_NORM,
    beta1=BETA1,
    beta2=BETA2,
    seed=DEFAULT_SEED,
):
    # Rewrite symbols with deduped, uppercase versions
    symbols = list(map(str.upper, set(symbols)))

    logging.info(f"model: {model_file}")
    logging.info(f"device: {device}")
    logging.info(f"existing_model: {existing_model}")
    logging.info(f"symbols: {symbols}")
    logging.info(f"refresh: {refresh}")
    logging.info(f"risk_neutral: {risk_neutral}")
    logging.info(f"only_embeddings: {only_embeddings}")
    logging.info(f"use_mixture: {use_mixture}")
    logging.info(f"window_size: {window_size}")
    logging.info(f"mixture_components: {mixture_components}")
    logging.info(f"feature_dimension: {feature_dimension}")
    logging.info(f"embedding_dimension: {embedding_dimension}")
    logging.info(f"gaussian_noise: {gaussian_noise}")
    logging.info(f"minibatch_size: {minibatch_size}")
    logging.info(f"dropout: {dropout}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"weight_decay: {weight_decay}")
    logging.info(f"use_batch_norm: {use_batch_norm}")
    logging.info(f"ADAM beta1: {beta1}")
    logging.info(f"ADAM beta2: {beta2}")
    logging.info(f"Seed: {seed}")

    model_network = embeddings = None
    if existing_model:
        model_network, embeddings, encoding = load_existing_model(
            existing_model, symbols
        )
        logging.info(f"Loaded model from file: {existing_model}")
    else:
        encoding = {s: i for i, s in enumerate(symbols)}

    logging.info(f"Encoding: {encoding}")

    data_store = stock_data.FileSystemStore("training_data")
    if use_hsmd:
        data_source = data_sources.HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = data_sources.YFinanceSource()

    history_loader = stock_data.CachingSymbolHistoryLoader(
        data_source, data_store, refresh
    )

    torch.random.manual_seed(seed)

    # Do split before any random weight initialization so that any
    # subsequent random number generator calls won't affect the split.
    # We want the splits to be the same for different architecutre
    # parameters to provide fair comparisons of different
    # architectures on the same split.

    train_loader, validation_loader = prepare_data(
        history_loader,
        symbols,
        encoding,
        window_size,
        minibatch_size=minibatch_size,
    )

    if model_network is None or embeddings is None:
        model_network, embeddings = create_new_model(
            embedding_size=len(symbols),
            window_size=window_size,
            mixture_components=mixture_components,
            feature_dimension=feature_dimension,
            embedding_dimension=embedding_dimension,
            gaussian_noise=gaussian_noise,
            use_mixture=use_mixture,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            risk_neutral=risk_neutral,
        )
        logging.info("Initialized new model")

    # Generate list of parameters we choose to train.
    # We always tune or train the embeddings:
    parameters = list(embeddings.parameters())

    # Add rest of model parameters unless we're training only the embeddings.
    if not only_embeddings:
        parameters.extend(model_network.parameters())

    logging.debug(f"parameters: {parameters}")

    # Define model, optimizer, loss function, and callbacks before calling train()
    model = architecture.ModelWithEmbedding(model_network, embeddings)
    model.to(device)

    optim = torch.optim.Adam(
        parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        eps=ADAM_EPSILON,
    )

    if use_mixture:
        loss_function = make_mixture_loss_function()
        validation_batch_callback = make_mixture_validation_batch_logger()
    else:
        loss_function = make_loss_function()
        validation_batch_callback = make_validation_batch_logger()

    epoch_callback = make_epoch_callback(model)
    loss_improvement_callback = make_loss_improvement_callback(
        model_file, only_embeddings, model, encoding, symbols
    )

    best_epoch, best_validation_loss, best_model = training.train(
        model=model,
        loss_function=loss_function,
        optim=optim,
        train_loader=train_loader,
        validation_loader=validation_loader,
        max_epochs=max_epochs,
        early_termination=early_termination,
        validation_batch_callback=validation_batch_callback,
        epoch_callback=epoch_callback,
        loss_improvement_callback=loss_improvement_callback,
    )
    logging.info(
        f"Training terminated. Best epoch: {best_epoch}; Best validation loss: {best_validation_loss}"
    )
    return best_epoch, best_validation_loss, best_model


@click.command()
@click.option(
    "--use-hsmd",
    default=None,
    show_default=True,
    help="Use huge stock market dataset if specified zip file (else use yfinance)",
)
@click.option(
    "--model",
    default="model.pt",
    show_default=True,
    help="Trained model output file.",
)
@click.option(
    "--existing-model",
    default=None,
    show_default=True,
    help="Existing model to load (for tuning).",
)
@click.option("--symbol", "-s", multiple=True, show_default=True)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    show_default=True,
    help="Refresh stock data",
)
@click.option(
    "--risk-neutral/--no-risk-neutral",
    is_flag=True,
    default=RISK_NEUTRAL,
    show_default=True,
    help="Use risk-neutral adjustment on predicted mean returns",
)
@click.option(
    "--only-embeddings",
    is_flag=True,
    default=False,
    show_default=True,
    help="Train only the embeddings",
)
@click.option(
    "--use-mixture/--no-mixture",
    is_flag=True,
    default=USE_MIXTURE,
    show_default=True,
    help="Use a mixture model?",
)
@click.option(
    "--early-termination",
    default=EARLY_TERMINATION,
    show_default=True,
    help="Terminate if no improvement in this number of iterations",
)
@click.option(
    "--learning-rate", default=OPT_LEARNING_RATE, show_default=True, type=float
)
@click.option("--dropout", default=OPT_DROPOUT, show_default=True, type=float)
@click.option(
    "--use-batch-norm/--no-use-batch-norm",
    is_flag=True,
    default=USE_BATCH_NORM,
    show_default=True,
)
@click.option(
    "--feature-dimension", default=OPT_FEATURE_DIMENSION, show_default=True, type=int
)
@click.option(
    "--mixture-components", default=OPT_MIXTURE_COMPONENTS, show_default=True, type=int
)
@click.option("--window-size", default=OPT_WINDOW_SIZE, show_default=True, type=int)
@click.option(
    "--embedding-dimension",
    default=OPT_EMBEDDING_DIMENSION,
    show_default=True,
    type=int,
)
@click.option(
    "--minibatch-size", default=OPT_MINIBATCH_SIZE, show_default=True, type=int
)
@click.option(
    "--gaussian-noise", default=OPT_GAUSSIAN_NOISE, show_default=True, type=float
)
@click.option("--weight-decay", default=OPT_WEIGHT_DECAY, show_default=True, type=float)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
def main_cli(
    use_hsmd,
    model,
    existing_model,
    symbol,
    refresh,
    risk_neutral,
    only_embeddings,
    use_mixture,
    early_termination,
    learning_rate,
    dropout,
    use_batch_norm,
    feature_dimension,
    mixture_components,
    window_size,
    embedding_dimension,
    minibatch_size,
    gaussian_noise,
    weight_decay,
    seed,
):
    run(
        use_hsmd,
        model_file=model,
        existing_model=existing_model,
        symbols=symbol,
        refresh=refresh,
        risk_neutral=risk_neutral,
        use_mixture=use_mixture,
        only_embeddings=only_embeddings,
        early_termination=early_termination,
        learning_rate=learning_rate,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        feature_dimension=feature_dimension,
        mixture_components=mixture_components,
        window_size=window_size,
        embedding_dimension=embedding_dimension,
        minibatch_size=minibatch_size,
        gaussian_noise=gaussian_noise,
        weight_decay=weight_decay,
        seed=seed,
    )


if __name__ == "__main__":
    main_cli()
