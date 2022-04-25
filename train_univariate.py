# Standard Python
import datetime as dt
import logging

import os.path
from typing import Iterable

# Common packages
import click
import numpy as np

import torch
import torch.utils.data
import torch.utils.data.dataloader

# Local imports
import data_sources
import stock_data
import mixture_model_stats
import time_series_datasets
import models
import architecture

logging.basicConfig(level=logging.INFO, force=True)

TRAIN_FRACTION = 0.80
SEED = 24  # 42

EPOCHS = 1000  # 30000
EARLY_TERMINATION = 100  # Was 1000

LEARNING_RATE = 0.00075 * 0.50  # 2
DROPOUT = 0.50
FEATURE_DIMENSION = 40
MIXTURE_COMPONENTS = 4  # Was 4, then 3
WINDOW_SIZE = 64
EMBEDDING_DIMENSION = 10  # Was 6
MINIBATCH_SIZE = 75  # 64
GAUSSIAN_NOISE = 0.0025
WEIGHT_DECAY = 5e-9  # 0.075

# These were optimized with hyperopt
OPT_LEARNING_RATE = 0.000689
OPT_DROPOUT = 0.130894
OPT_FEATURE_DIMENSION = 86
OPT_MIXTURE_COMPONENTS = 3
OPT_WINDOW_SIZE = 256
OPT_EMBEDDING_DIMENSION = 3
OPT_MINIBATCH_SIZE = 248
OPT_GAUSSIAN_NOISE = 0.000226
OPT_WEIGHT_DECAY = 8.489603e-07


# Following parameters haven't been optimized yet.

BETA1 = 0.95
BETA2 = 0.999
ADAM_EPSILON = 1e-8  # 1e-5
USE_BATCH_NORM = False  # False
ACTIVATION = torch.nn.ReLU()
MAX_GRADIENT_NORM = 1.0

null_model_loss = float("inf")

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_ROOT = os.path.join(ROOT_PATH, "models")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def load_or_create_model(
    window_size=WINDOW_SIZE,
    mixture_components=MIXTURE_COMPONENTS,
    feature_dimension=FEATURE_DIMENSION,
    embedding_dimension=EMBEDDING_DIMENSION,
    gaussian_noise=GAUSSIAN_NOISE,
    model_file=None,
    use_batch_norm=USE_BATCH_NORM,
    dropout=DROPOUT,
):
    default_network_class = architecture.MixtureModel

    try:
        model = torch.load(model_file)
        network = model.network
        logging.info(f"Loaded model from file: {model_file}")

    except Exception:
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
        )
        logging.info("Initialized new model")

    parameters = list(network.parameters())

    return network, parameters


def prepare_data(
    symbol_list: Iterable[str],
    window_size: int,
    refresh: bool = False,
    minibatch_size: int = MINIBATCH_SIZE,
):
    # Refresh historical data
    logging.info("Reading historical data")
    splits_by_symbol = {}
    symbol_encoding = {}

    data_store = stock_data.FileSystemStore(os.path.join(ROOT_PATH, "training_data"))
    data_source = data_sources.YFinanceSource()
    history_loader = stock_data.CachingSymbolHistoryLoader(data_source, data_store)

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

    for i, s in enumerate(symbol_list):
        symbol_encoding[s] = i

        # history_loader can load many symbols at once for multivariate
        # but here we just load the single symbol of interest.  Since we expect
        # just one dataframe, grab it with next() instead of using a combiner()
        # (see stock-data.py).)
        symbol_history = next(history_loader(s, overwrite_existing=refresh))[1]

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
            time_series_datasets.ContextWindowEncodingAndTarget(i, symbol_dataset)
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
    test_dataset = torch.utils.data.ConcatDataset(
        [splits_by_symbol[s]["test"] for s in symbol_list]
    )

    train_dataloader = torch.utils.data.dataloader.DataLoader(
        train_dataset, batch_size=minibatch_size, drop_last=True, shuffle=True
    )

    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=len(test_dataset), drop_last=True, shuffle=True
    )

    return symbol_encoding, train_dataloader, test_dataloader


def make_loss_function():
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


def log_mean_error(output, target):
    log_p, mu = output[:2]
    mb_size, components, channels = mu.shape
    combined_mu = torch.sum(
        mu * torch.exp(log_p).unsqueeze(2).expand((mb_size, components, channels)),
        dim=1,
    )
    mean_error = torch.mean(target.squeeze(2) - combined_mu, dim=0)
    logging.info(f"mean_error: {mean_error}")


def make_test_batch_logger():
    def log_epoch(epoch, batch, output, target, loss):
        if output:
            log_p, mu, inv_sigma = output[:3]
            logging.info(
                f"last epoch p:\n{torch.exp(log_p)[:6].detach().cpu().numpy()}"
            )
            logging.info(f"last epoch mu:\n{mu[:6].detach().cpu().numpy()}")
            logging.info(f"last epoch sigma:\n{inv_sigma[:6].detach().cpu().numpy()}")

            log_mean_error(output, target)

    return log_epoch


def make_save_model(model_root, just_embeddings, model, encoding, symbols):
    os.makedirs(model_root, exist_ok=True)

    def save_model(epoch, epoch_loss, prefix=""):
        wrapped_model = models.StockModelV2(
            network=model,
            symbols=symbols,
            epochs=epoch,
            date=dt.datetime.now(),
            null_model_loss=null_model_loss,
            loss=epoch_loss,
        )

        if not just_embeddings:
            torch.save(wrapped_model, os.path.join(model_root, f"{prefix}model.pt"))

        torch.save(
            encoding,
            os.path.join(model_root, f"{prefix}symbol_encodings.pt"),
        )

    return save_model


def make_model_improvement_callback(
    model_root, just_embeddings, model, encoding, symbols
):
    save_model = make_save_model(model_root, just_embeddings, model, encoding, symbols)

    def model_improvement_callback(epoch, epoch_loss):
        save_model(epoch, epoch_loss)

    return model_improvement_callback


def make_epoch_callback(model_root, just_embeddings, model, encoding, symbols):
    save_model = make_save_model(model_root, just_embeddings, model, encoding, symbols)

    def epoch_callback(epoch, train_epoch_loss, test_epoch_loss):
        logging.debug(f"parameters: {(list(model.embedding.parameters()))}")
        save_model(epoch, test_epoch_loss, prefix="last_")

    return epoch_callback


def do_batches(epoch, model, data_loader, loss_function, optim, training, callback):
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


def run(
    model_file,
    symbols,
    refresh,
    tune_embeddings,
    just_embeddings,
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
    max_epochs=EPOCHS,
    early_termination=EARLY_TERMINATION,
    model_root=MODEL_ROOT,
):
    logging.debug(f"model_root: {model_root}")
    logging.debug(f"device: {device}")

    # Rewrite symbols in `symbol` with uppercase versions
    symbols = list(map(str.upper, set(symbols)))

    logging.info(f"model_file: {model_file}")
    logging.info(f"symbols: {symbols}")
    logging.info(f"refresh: {refresh}")
    logging.info(f"window_size: {window_size}")
    logging.info(f"mixture_components: {mixture_components}")
    logging.info(f"feature_dimension: {feature_dimension}")
    logging.info(f"embedding_dimension: {embedding_dimension}")
    logging.info(f"gaussian_noise: {gaussian_noise}")
    logging.info(f"minibatch_size: {minibatch_size}")
    logging.info(f"dropout: {dropout}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"weight_decay: {weight_decay}")

    logging.info(f"Seed: {SEED}")
    torch.random.manual_seed(SEED)

    model_network, parameters = load_or_create_model(
        window_size=window_size,
        mixture_components=mixture_components,
        feature_dimension=feature_dimension,
        embedding_dimension=embedding_dimension,
        gaussian_noise=gaussian_noise,
        model_file=model_file,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
    )
    model_network = model_network.to(device)

    embeddings = torch.nn.Embedding(len(symbols), embedding_dimension)
    embeddings = embeddings.to(device)

    if tune_embeddings:
        old_embeddings = next(torch.load(tune_embeddings).parameters())
        mean_embedding = old_embeddings.mean(dim=0)
        new_embeddings = next(embeddings.parameters())
        n, m = new_embeddings.shape
        new_embeddings.data = mean_embedding.unsqueeze(0).expand(n, m).clone()

    parameters = list(embeddings.parameters())

    if just_embeddings:
        model_network.eval()
    else:
        model_network.train()
        parameters.extend(model_network.parameters())

    logging.debug(f"parameters: {parameters}")

    encoding, train_loader, test_loader = prepare_data(
        symbols, window_size, refresh, minibatch_size=minibatch_size
    )
    the_model = architecture.ModelWithEmbedding(model_network, embeddings)

    optim = torch.optim.Adam(
        parameters,
        lr=learning_rate,
        eps=ADAM_EPSILON,
        betas=(BETA1, BETA2),
        weight_decay=weight_decay,
    )

    # Initialize state for early termination monitoring
    best_test_loss = float("inf")
    best_epoch = -1

    loss_function = make_loss_function()
    train_batch_callback = lambda epoch, batch, output, target, loss: None
    test_batch_callback = make_test_batch_logger()
    epoch_callback = make_epoch_callback(
        model_root, just_embeddings, the_model, encoding, symbols
    )
    model_improvement_callback = make_model_improvement_callback(
        model_root, just_embeddings, the_model, encoding, symbols
    )

    # This is the main epoch loop
    for epoch in range(max_epochs):

        train_epoch_loss = do_batches(
            epoch,
            the_model,
            train_loader,
            loss_function,
            optim,
            True,
            train_batch_callback,
        )

        # Evalute the loss on the test set
        # Don't compute gradients
        with torch.no_grad():
            test_epoch_loss = do_batches(
                epoch,
                the_model,
                test_loader,
                loss_function,
                optim,
                False,
                test_batch_callback,
            )

        epoch_callback(epoch, train_epoch_loss, test_epoch_loss)
        logging.info(f"    Epoch {epoch}: loss (train): {train_epoch_loss:.4f}")

        if test_epoch_loss < best_test_loss:
            best_test_loss = test_epoch_loss
            best_epoch = epoch
            flag = "**"

            model_improvement_callback(epoch, test_epoch_loss)
        else:
            flag = "  "
        logging.info(
            f" {flag} Epoch {epoch}: loss (test): {test_epoch_loss:.4f}  best epoch: {best_epoch}  best loss:{best_test_loss:.4f} {flag}"
        )
        if epoch >= best_epoch + early_termination:
            logging.info(
                f"No improvement in {early_termination} epochs.  Terminating early."
            )
            break  # Terminate early

    return best_test_loss


@click.command()
@click.option(
    "--model_file",
    default=None,
    show_default=True,
    help="Output file name for trained model",
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
    "--tune_embeddings", show_default=True, help="Load existing embedding file"
)
@click.option(
    "--just_embeddings",
    is_flag=True,
    default=False,
    show_default=True,
    help="Train only the embeddings",
)
@click.option(
    "--learning_rate", default=OPT_LEARNING_RATE, show_default=True, type=float
)
@click.option("--dropout", default=OPT_DROPOUT, show_default=True, type=float)
@click.option(
    "--feature_dimension", default=OPT_FEATURE_DIMENSION, show_default=True, type=int
)
@click.option(
    "--mixture_components", default=OPT_MIXTURE_COMPONENTS, show_default=True, type=int
)
@click.option("--window_size", default=OPT_WINDOW_SIZE, show_default=True, type=int)
@click.option(
    "--embedding_dimension",
    default=OPT_EMBEDDING_DIMENSION,
    show_default=True,
    type=int,
)
@click.option(
    "--minibatch_size", default=OPT_MINIBATCH_SIZE, show_default=True, type=int
)
@click.option(
    "--gaussian_noise", default=OPT_GAUSSIAN_NOISE, show_default=True, type=float
)
@click.option("--weight_decay", default=OPT_WEIGHT_DECAY, show_default=True, type=float)
@click.option("--model_root", default=MODEL_ROOT, show_default=True)
def main_cli(
    model_file,
    symbol,
    refresh,
    tune_embeddings,
    just_embeddings,
    learning_rate,
    dropout,
    feature_dimension,
    mixture_components,
    window_size,
    embedding_dimension,
    minibatch_size,
    gaussian_noise,
    weight_decay,
    model_root,
):
    run(
        model_file=model_file,
        symbols=symbol,
        refresh=refresh,
        tune_embeddings=tune_embeddings,
        just_embeddings=just_embeddings,
        learning_rate=learning_rate,
        dropout=dropout,
        feature_dimension=feature_dimension,
        mixture_components=mixture_components,
        window_size=window_size,
        embedding_dimension=embedding_dimension,
        minibatch_size=minibatch_size,
        gaussian_noise=gaussian_noise,
        weight_decay=weight_decay,
        model_root=model_root,
        use_batch_norm=USE_BATCH_NORM,
    )


if __name__ == "__main__":
    main_cli()
