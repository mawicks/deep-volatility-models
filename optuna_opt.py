import logging

import optuna

import train_univariate

logging.basicConfig(level=logging.INFO)

SYMBOLS = [
    "bnd",
    "edv",
    "tyd",
    "gld",
    "vnq",
    "vti",
    "spy",
    "qqq",
    "qld",
    "xmvm",
    "vbk",
    "xlv",
    "fxg",
    "rxl",
    "fxl",
    "ibb",
    "vgt",
    "iyf",
    "xly",
    "uge",
    "jnk",
]


def objective(trial):
    mixture_components = trial.suggest_int("mixture_components", 1, 5)
    feature_dimension = trial.suggest_int("feature_dimension", 5, 50)
    embedding_dimension = trial.suggest_int("embedding_dimension", 3, 15)
    gaussian_noise = trial.suggest_float("gaussian_noise", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.75)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-5, log=True)
    window_size = trial.suggest_categorical("window_size", [16, 64, 256])
    minibatch_size = trial.suggest_int("minibatch_size", 32, 256)

    logging.info("************************")
    logging.info(f"mixture_components: {mixture_components}")
    logging.info(f"feature_dimension: {feature_dimension}")
    logging.info(f"embedding_dimension: {embedding_dimension}")
    logging.info(f"gaussian_noise: {gaussian_noise}")
    logging.info(f"dropout: {dropout}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"weight_decay: {weight_decay}")
    logging.info(f"window_size: {window_size}")
    logging.info(f"minibatch_size: {minibatch_size}")

    loss = train_univariate.run(
        existing_model=None,
        symbols=SYMBOLS,
        refresh=False,
        only_embeddings=False,
        window_size=window_size,
        mixture_components=mixture_components,
        feature_dimension=feature_dimension,
        gaussian_noise=gaussian_noise,
        embedding_dimension=embedding_dimension,
        minibatch_size=minibatch_size,
        use_batch_norm=False,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    logging.info(f"loss: {loss}")
    logging.info("************************")

    return loss


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=400)
    logging.info(f"{study.best_params}")
