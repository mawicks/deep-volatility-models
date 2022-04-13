# Standard Python
import datetime as dt
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

TRAIN_FRACTION = 0.80
SEED = 24  # 42

EPOCHS = 30000
EPOCH_SHOW_PROGRESS_INTERVAL = 10
DEFAULT_WINDOW_SIZE = 64
EMBEDDING_DIMENSION = 10  # Was 6
MINIBATCH_SIZE = 75  # 64
FEATURE_DIMENSION = 40
DEFAULT_MIXTURE_COMPONENTS = 4  # Was 4, then 3
DROPOUT_P = 0.50
BETA1 = 0.95
BETA2 = 0.999
ADAM_EPSILON = 1e-8  # 1e-5
USE_BATCH_NORM = False  # False
ACTIVATION = torch.nn.ReLU()
MAX_GRADIENT_NORM = 1.0

LR = 0.00075 * 0.50  # 2
WEIGHT_DECAY = 5e-9  # 0.075

null_model_loss = float("inf")

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_ROOT = os.path.join(ROOT_PATH, "models")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def load_or_create_model(
    window_size=DEFAULT_WINDOW_SIZE,
    mixture_components=DEFAULT_MIXTURE_COMPONENTS,
    model_file=None,
):
    default_network_class = architecture.MixtureModel

    try:
        model = torch.load(model_file)
        network = model.network
        print(f"Loaded model from file: {model_file}")

    except Exception:
        network = default_network_class(
            window_size,
            1,
            feature_dimension=FEATURE_DIMENSION,
            mixture_components=mixture_components,
            exogenous_dimension=EMBEDDING_DIMENSION,
            dropout=DROPOUT_P,
            use_batch_norm=USE_BATCH_NORM,
            activation=ACTIVATION,
        )
        print("Initialized new model")

    parameters = list(network.parameters())

    return network, parameters


def prepare_data(symbol_list: Iterable[str], window_size: int, refresh: bool = False):
    # Refresh historical data
    print("Reading historical data")
    splits_by_symbol = {}
    symbol_encoding = {}

    data_store = stock_data.FileSystemStore(os.path.join(ROOT_PATH, "training_data"))
    data_source = data_sources.YFinanceSource()
    history_loader = stock_data.CachingSymbolHistoryLoader(data_source, data_store)

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
            symbol_history.log_return,
            1 + window_size,
            create_channel_dim=True,
        )
        print("windowed_returns[0]: ", windowed_returns[0])
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
        train_dataset, batch_size=MINIBATCH_SIZE, drop_last=True, shuffle=True
    )

    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=len(test_dataset), drop_last=True, shuffle=False
    )

    return symbol_encoding, train_dataloader, test_dataloader


def batch_output(model_network, embeddings, batch):
    (window, encoding), true_values = batch
    window = window.to(device)
    encoding = encoding.to(device)
    true_values = true_values.to(device)
    symbol_embedding = embeddings(encoding)

    log_p, mu, inv_sigma = model_network((window, symbol_embedding))[:3]

    return log_p, mu, inv_sigma, true_values


def make_loss_function(model_network, embeddings, device):
    def loss_function(batch):
        log_p, mu, inv_sigma, true_values = batch_output(
            model_network, embeddings, batch
        )

        mb_size, components, channels = mu.shape
        combined_mu = torch.sum(
            mu * torch.exp(log_p).unsqueeze(2).expand((mb_size, components, channels)),
            dim=1,
        )

        mean_error = torch.mean(true_values.squeeze(2) - combined_mu, dim=0)

        loss = -torch.mean(
            mixture_model_stats.multivariate_log_likelihood(
                true_values.squeeze(2), log_p, mu, inv_sigma
            )
        )

        if np.isnan(float(loss)):
            print("log_p: ", log_p)
            print("mu: ", mu)
            print("inv_sigma: ", inv_sigma)
            for p in model_network.parameters():
                print("parameter", p)
                print("gradient", p.grad)
            raise Exception("Got a nan")
        return loss, mean_error

    return loss_function


def make_batch_logger(model_network, embeddings):
    def log_batch_info(batch):
        if batch:
            log_p, mu, inv_sigma = batch_output(model_network, embeddings, batch)[:3]
            print("last batch p:\n", torch.exp(log_p)[:6].detach().cpu().numpy())
            print("last batch mu:\n", mu[:6].detach().cpu().numpy())
            print("last batch sigma:\n", inv_sigma[:6].detach().cpu().numpy())

    return log_batch_info


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
@click.option("--tune_embeddings", help="Load existing embedding file")
@click.option(
    "--just_embeddings",
    is_flag=True,
    default=False,
    show_default=True,
    help="Train only the embeddings",
)
@click.option("--window_size", default=DEFAULT_WINDOW_SIZE, type=int)
@click.option("--mixture_components", default=DEFAULT_MIXTURE_COMPONENTS, type=int)
def main(
    model_file,
    symbol,
    refresh,
    tune_embeddings,
    just_embeddings,
    window_size,
    mixture_components,
):
    print(f"model_root: {MODEL_ROOT}")
    print(f"device: {device}")

    # Rewrite symbols in `symbol` with uppercase versions
    symbol = list(map(str.upper, symbol))

    print(f"model_file: {model_file}")
    print(f"symbol: {symbol}")
    print(f"refresh: {refresh}")
    print(f"window_sizet: {window_size}")

    print(f"Seed: {SEED}")
    torch.random.manual_seed(SEED)

    model_network, parameters = load_or_create_model(
        window_size=window_size,
        mixture_components=mixture_components,
        model_file=model_file,
    )
    model_network = model_network.to(device)

    embeddings = torch.nn.Embedding(len(symbol), EMBEDDING_DIMENSION)
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

    print(parameters)

    encoding, train_loader, test_loader = prepare_data(symbol, window_size, refresh)

    optim = torch.optim.Adam(
        parameters,
        lr=LR,
        eps=ADAM_EPSILON,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    best_test_loss = float("inf")
    best_epoch = -1

    # Next line is here mainly to keep linter happy.
    log_p = mu = inv_sigma = torch.tensor([])

    loss_function = make_loss_function(model_network, embeddings, device)
    batch_logger = make_batch_logger(model_network, embeddings)

    for e in range(EPOCHS):
        batch_lossees_train = []

        model_network.train()

        batch = None
        for batch in train_loader:

            loss = loss_function(batch)[0]

            optim.zero_grad()

            # To debug Nans, uncomment following line:
            # with torch.autograd.detect_anomaly():
            loss.backward()

            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters, MAX_GRADIENT_NORM, norm_type=float("inf")
            )
            optim.step()

            batch_lossees_train.append(float(loss))

        if e % EPOCH_SHOW_PROGRESS_INTERVAL == 0:
            batch_logger(batch)

        print(list(embeddings.parameters()))

        epoch_loss_train = float(np.mean(batch_lossees_train))
        print(f"epoch {e} train loss: {epoch_loss_train:.4f}")

        # Evalute the loss on the test set
        batch_losses_test = []
        batch_mean_errors_test = []

        # Don't compute gradients
        with torch.no_grad():
            model_network.eval()

            for batch in test_loader:
                batch_loss_test, epoch_mean_error_test = loss_function(batch)
                batch_losses_test.append(float(batch_loss_test))
                batch_mean_errors_test.append(float(epoch_mean_error_test))

            epoch_loss_test = float(np.mean(batch_losses_test))
            epoch_mean_error_test = float(np.mean(batch_mean_errors_test))

            model = models.StockModelV2(
                network=model_network,
                symbols=symbol,
                epochs=e,
                date=dt.datetime.now(),
                null_model_loss=null_model_loss,
                loss=epoch_loss_test,
            )
            if epoch_loss_test < best_test_loss:
                best_test_loss = epoch_loss_test
                best_epoch = e
                flag = "**"

                if not just_embeddings:
                    torch.save(model, os.path.join(MODEL_ROOT, "embedding_model.pt"))

                torch.save(embeddings, os.path.join(MODEL_ROOT, "embeddings.pt"))
                torch.save(
                    encoding,
                    os.path.join(MODEL_ROOT, "symbol_encodings.pt"),
                )
            else:
                flag = "  "
            print(
                f" {flag} epoch loss (test): {epoch_loss_test:.4f}  best epoch: {best_epoch}  best loss:({best_test_loss:.4f})) {flag}"
            )
            print(f"    mean error (test): {epoch_mean_error_test:.7f}")

            torch.save(model, os.path.join(MODEL_ROOT, "last_embedding_model.pt"))
            torch.save(embeddings, os.path.join(MODEL_ROOT, "last_embeddings.pt"))
            torch.save(
                encoding,
                os.path.join(MODEL_ROOT, "last_symbol_encoding.pt"),
            )


if __name__ == "__main__":
    main()
