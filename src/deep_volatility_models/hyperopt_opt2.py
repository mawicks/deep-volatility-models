import logging

from hyperopt import hp, tpe, fmin, Trials
import numpy as np

import deep_volatility_models.train_univariate as train_univariate

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
    "aal",
    "amd",
    "amzn",
    "bac",
    "cmcsa",
    "cmg",
    "dis",
    "f",
    "fb",
    "ge",
    "gld",
    "gme",
    "goog",
    "iyr",
    "jnk",
    "mro",
    "nflx",
    "qqq",
    "sbux",
    "spy",
    "t",
    "trip",
    "twtr",
    "v",
    "wfc",
    "vti",
    "ba",
    "c",
    "gm",
    "intc",
    "jpm",
    "hpe",
    "ko",
    "kr",
    "mgm",
    "msft",
    "mvis",
    "oxy",
    "pins",
    "uber",
    "x",
    "xom",
    "gps",
    "jnj",
    "nke",
    "pypl",
    "wmt",
    "ups",
    "baba",
    "sq",
    "fdx",
    "snap",
    "amc",
    "pfe",
    "rkt",
    "aapl",
    "pton",
    "csco",
    "roku",
    "sq",
    "snow",
    "bnd",
    "vbk",
    "xmvm",
    "nvda",
    "vz",
]

# Dedup
SYMBOLS = list(set(SYMBOLS))

search_space = {
    "mixture_components": 3 + hp.randint("mixture_components", 3),
    "feature_dimension": 50 + hp.randint("feature_dimension", 41),
    "embedding_dimension": 3 + hp.randint("embedding_dimension", 8),
    "gaussian_noise": hp.loguniform("gaussian_noise", np.log(1e-4), np.log(1e-3)),
    "dropout": hp.uniform("dropout", 0.075, 0.125),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-3)),
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-7), np.log(1e-6)),
    "window_size": hp.choice("window_size", [64, 256]),
    "minibatch_size": 128 + hp.randint("minibatch_size", 129),
}


def objective(parameters):
    # Be a good citizen and make a copy since we're going to modify the dictionary
    parameters = parameters.copy()

    # `minibatch_size` has to be a Python int, not a numpy int.
    parameters["minibatch_size"] = int(parameters["minibatch_size"])

    logging.info("************************")
    for key, value in parameters.items():
        logging.info(f"{key}: {value}")

    loss = train_univariate.run(
        existing_model=None,
        symbols=SYMBOLS,
        refresh=False,
        only_embeddings=False,
        max_epochs=400,
        early_termination=20,
        **parameters,
    )

    logging.info(f"loss: {loss}")
    logging.info("************************")

    return loss


if __name__ == "__main__":
    trials = Trials()

    best = fmin(
        objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
    )
    print(trials.trials)

    print("\n***** Best parameters *****")

    print(best)
