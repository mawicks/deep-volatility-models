# Standard Python
import copy
import datetime as dt
import os.path

# Common packages
import click
import numpy as np

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

# Local imports
import data_sources
import price_history
import mixture_model_stats
import time_series
import models
import architecture

TRAIN_FRACTION = 0.80
SEED = 24  # 42

EPOCHS = 30000
DEFAULT_WINDOW_SIZE = 64
EMBEDDING_DIMENSION = 10  # Was 6
MINIBATCH_SIZE = 75  # 64
FEATURE_DIMENSION = 40
DEFAULT_MIXTURE_COMPONENTS = 3  # Was 4, then 3
DROPOUT_P = 0.50
BETA1 = 0.95
BETA2 = 0.999
ADAM_EPSILON = 1e-8  # 1e-5
USE_BATCH_NORM = False  # False
MAX_GRADIENT_NORM = 1.0
ACTIVATION = torch.nn.ReLU()  # torch.nn.Softplus()

LR = 0.00075 * 0.50  # 2
WEIGHT_DECAY = 5e-9  # 0.075

null_model_loss = float("inf")

root = os.path.expanduser("~/Google Drive")
# data_root = os.path.join(root, "Projects/data")
current_path = os.path.dirname(os.path.realpath(__file__))
data_root = os.path.join(current_path, "data")
model_root = os.path.join(current_path, "models")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

training_start_date = dt.date(year=1998, month=1, day=1)
test_size = 756

sigmoid = torch.nn.Sigmoid()
logsoftmax = torch.nn.LogSoftmax(dim=1)


def get_model(
    window_size=DEFAULT_WINDOW_SIZE,
    mixture_components=DEFAULT_MIXTURE_COMPONENTS,
    model_file=None,
):
    default_network_class = architecture.MixtureModel

    try:
        model = torch.load(model_file)
        n_network = model.network
        print(f"Loaded model from file: {model_file}")

    except Exception:
        n_network = default_network_class(
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

    parameters = list(n_network.parameters())

    return n_network, parameters


class SymbolDataset(torch.utils.data.Dataset):
    def __init__(self, symbol_encoding, symbol_dataset):
        self.__symbol_encoding = symbol_encoding
        self.__symbol_dataset = symbol_dataset

    def __len__(self):
        return len(self.__symbol_dataset)

    def __getitem__(self, i):
        context, target = self.__symbol_dataset[i]
        return self.__symbol_encoding, context, target


@click.command()
@click.option("--project", default="None")
@click.option(
    "--model_file",
    default=None,
    show_default=True,
    help="File name of resulting model",
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
    project,
    model_file,
    symbol,
    refresh,
    tune_embeddings,
    just_embeddings,
    window_size,
    mixture_components,
):
    print("data_root: ", data_root)
    print("model_root: ", model_root)
    print("device: ", device)

    # Rewrite symbosl in `symbol` with uppercase versions
    symbol = list(map(str.upper, symbol))

    print("model_file: ", model_file)
    print("symbol: ", symbol)
    print("refresh: ", refresh)
    print("context: ", window_size)

    print(f"Seed: {SEED}")
    torch.random.manual_seed(SEED)

    n_network, parameters = get_model(
        window_size=window_size,
        mixture_components=mixture_components,
        model_file=model_file,
    )
    n_network = n_network.to(device)

    embeddings = torch.nn.Embedding(len(symbol), EMBEDDING_DIMENSION)

    if tune_embeddings:
        old_embeddings = next(torch.load(tune_embeddings).parameters())
        mean_embedding = old_embeddings.mean(dim=0)
        new_embeddings = next(embeddings.parameters())
        n, m = new_embeddings.shape
        new_embeddings.data = mean_embedding.unsqueeze(0).expand(n, m).clone()

    embeddings = embeddings.to(device)

    parameters = list(embeddings.parameters())

    if just_embeddings:
        n_network.eval()
    else:
        n_network.train()
        parameters.extend(n_network.parameters())

    print(parameters)

    window_size = n_network.window_size

    # Refresh historical data
    print("Reading historical data")
    splits_by_symbol = {}
    symbol_encoding_dict = {}

    data_store = price_history.FileSystemStore("training_data")
    data_source = data_sources.YFinanceSource()
    history_loader = price_history.CachingSymbolHistoryLoader(data_source, data_store)

    for i, s in enumerate(symbol):
        symbol_encoding_dict[s] = i

        symbol_history = history_loader(s, overwrite_existing=refresh)
        log_returns = symbol_history.loc[:, (s, "log_return")]
        windowed_returns = time_series.RollingWindowSeries(
            log_returns,
            1 + window_size,
            create_channel_dim=True,
        )
        print(windowed_returns[0])
        symbol_dataset = time_series.ContextAndTargetSeries(windowed_returns, 1)
        _cov, _target = symbol_dataset[0]
        dataset_with_target = SymbolDataset(i, symbol_dataset)

        _label, _cov, _target = dataset_with_target[0]
        train_size = int(TRAIN_FRACTION * len(dataset_with_target))
        lengths = [train_size, len(dataset_with_target) - train_size]
        train, test = torch.utils.data.random_split(dataset_with_target, lengths)
        splits_by_symbol[s] = {"train": train, "test": test}

    train_dataset = torch.utils.data.ConcatDataset(
        [splits_by_symbol[s]["train"] for s in symbol]
    )
    test_dataset = torch.utils.data.ConcatDataset(
        [splits_by_symbol[s]["test"] for s in symbol]
    )

    training_dataloader = torch.utils.data.dataloader.DataLoader(
        train_dataset, batch_size=MINIBATCH_SIZE, drop_last=True, shuffle=True
    )

    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=len(test_dataset), drop_last=True, shuffle=False
    )

    channels = 1
    optim = torch.optim.Adam(
        parameters,
        lr=LR,
        eps=ADAM_EPSILON,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    best_test_loss = float("inf")

    best_epoch = -1
    log_p = mu = inv_sigma = torch.tensor(
        []
    )  # This is here mainly to keep type checker and linter happy.

    for e in range(EPOCHS):
        epoch_losses = []

        for minibatch, (symbol_encoding, window_size, true_values) in enumerate(
            training_dataloader
        ):
            window_size = window_size.to(device)
            symbol_encoding = symbol_encoding.to(device)
            true_values = true_values.to(device)

            symbol_embedding = embeddings(symbol_encoding)
            log_p, mu, inv_sigma = n_network(window_size, symbol_embedding)[:3]

            mb_size, components, channels = mu.shape
            combined_mu = torch.sum(
                mu
                * torch.exp(log_p).unsqueeze(2).expand((mb_size, components, channels)),
                dim=1,
            )

            # Note that bias_error is computed on the entire mini-batch and then squared
            # It is not the usual MSE. It is square of the mean of the error, not mean of the square of the error.
            mean_error = torch.mean(true_values.squeeze(2) - combined_mu, dim=0)
            # print('\n\ttrain mean true_values', torch.mean(true_values.squeeze(2), dim=0))
            # print('\ttrain mean combined_mu', torch.mean(combined_mu, dim=0))
            # print('\ttrain mean_error: ', mean_error)
            bias_error = torch.mean(mean_error ** 2)

            loss = -torch.mean(
                mixture_model_stats.multivariate_log_likelihood(
                    true_values.squeeze(2), log_p, mu, inv_sigma
                )
            )

            if np.isnan(float(loss)):
                print("log_p: ", log_p)
                print("mu: ", mu)
                print("inv_sigma: ", inv_sigma)
                for p in n_network.parameters():
                    print("parameter", p)
                    print("gradient", p.grad)
                raise Exception("Got a nan")

            # print(mu, log_sigma, log_p, true_values)
            optim.zero_grad()

            # To debug Nans, uncomment following line:
            # with torch.autograd.detect_anomaly():
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters, MAX_GRADIENT_NORM, norm_type=float("inf")
            )
            optim.step()

            epoch_losses.append(float(loss))
            # epoch_p = torch.exp(log_p)
            # batch_mus = torch.sum(epoch_p.unsqueeze(2) * mu,
            # dim=1).detach().numpy()

        if e % 10 == 0:
            print("last batch p:\n", torch.exp(log_p)[:6].detach().cpu().numpy())
            print("last batch mu:\n", mu[:6].detach().cpu().numpy())
            print("last batch sigma:\n", inv_sigma[:6].detach().cpu().numpy())

        if e % 1 == 0:
            print(list(embeddings.parameters()))

            train_loss = float(np.mean(epoch_losses))
            print("epoch {} train loss: {:.4f}".format(e, train_loss))

            # print(
            # '\tepoch sigma(mean) (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
            #       .format(
            #        np.min(epoch_sigmas), np.mean(epoch_sigmas),
            #        np.max(epoch_sigmas)))
            # print('\tepoch mu (min/mean/max): {:.4f}/{:.4f}/{:.4f}'.format(
            #    np.min(epoch_mus), np.mean(epoch_mus), np.max(epoch_mus)))

            # Evalute the loss on the test set
            test_epoch_losses = []
            test_epoch_bias_errors = []

            # Don't compute gradiets
            with torch.no_grad():
                # Use a copy of the model for evaluation so there's no
                # possibility of leaking test data into the model.
                # Use the model in eval() mode.
                network_copy = copy.deepcopy(n_network).eval()

                for minibatch, (symbol_encoding, window_size, true_values) in enumerate(
                    test_dataloader
                ):
                    window_size = window_size.to(device)
                    true_values = true_values.to(device)
                    symbol_encoding = symbol_encoding.to(device)

                    symbol_embedding = embeddings(symbol_encoding)
                    log_p, mu, inv_sigma = network_copy(window_size, symbol_embedding)[
                        :3
                    ]

                    mb_size, components, channels = mu.shape
                    combined_mu = torch.sum(
                        mu
                        * torch.exp(log_p)
                        .unsqueeze(2)
                        .expand((mb_size, components, channels)),
                        dim=1,
                    )

                    # Note that bias_error is computed on the entire mini-batch and then squared
                    # It is not the usual MSE. It is square of the mean of the error, not mean of the square of the error.
                    mean_error = torch.mean(true_values.squeeze(2) - combined_mu, dim=0)
                    print(
                        "\n\ttest mean true_values",
                        torch.mean(true_values.squeeze(2), dim=0),
                    )
                    print("\ttest mean combined_mu", torch.mean(combined_mu, dim=0))
                    print("\ttest mean_error: ", mean_error)
                    bias_error = torch.mean(mean_error ** 2)

                    log_loss = -torch.mean(
                        mixture_model_stats.multivariate_log_likelihood(
                            true_values.squeeze(2), log_p, mu, inv_sigma
                        )
                    )

                    test_epoch_losses.append(float(log_loss))
                    test_epoch_bias_errors.append(float(bias_error))

                test_loss = float(np.mean(test_epoch_losses))
                test_bias_error = float(np.mean(test_epoch_bias_errors))

                model = models.StockModelV2(
                    network=network_copy,
                    symbols=symbol,
                    epochs=e,
                    date=dt.datetime.now(),
                    null_model_loss=null_model_loss,
                    loss=test_loss,
                )
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = e
                    flag = " ** "

                    if not just_embeddings:
                        torch.save(
                            model, os.path.join(model_root, "embedding_model.pt")
                        )

                    torch.save(embeddings, os.path.join(model_root, "embeddings.pt"))
                    torch.save(
                        symbol_encoding_dict,
                        os.path.join(model_root, "symbol_encodings.pt"),
                    )
                else:
                    flag = ""
                print(
                    "\t     test log loss: {:.4f}  test bias error: {:.7f}".format(
                        test_loss, test_bias_error
                    )
                )
                print(
                    "\t{}total test loss: {:.4f} ({} {:.4f}/{:.4f}){}".format(
                        flag,
                        test_loss,
                        best_epoch,
                        best_test_loss,
                        -null_model_loss,
                        flag,
                    )
                )

                torch.save(model, os.path.join(model_root, "last_embedding_model.pt"))
                torch.save(embeddings, os.path.join(model_root, "last_embeddings.pt"))
                torch.save(
                    symbol_encoding_dict,
                    os.path.join(model_root, "last_symbol_encoding.pt"),
                )


if __name__ == "__main__":
    main()
