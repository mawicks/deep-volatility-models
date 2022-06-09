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
    "amzn",
    "bac",
    "cmcsa",
    "cmg",
    "dis",
    "f",
    "fb",
    "gld",
    "gme",
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
    "nvda",
    "vz",
]

search_space = {
    "feature_dimension": 30 + hp.randint("feature_dimension", 61),
    "embedding_dimension": 2 + hp.randint("embedding_dimension", 6),
    "gaussian_noise": hp.loguniform("gaussian_noise", np.log(1e-4), np.log(1e-2)),
    "dropout": hp.uniform("dropout", 0, 0.01),
    "learning_rate": hp.loguniform("learning_rate", np.log(4e-4), np.log(2e-3)),
    "weight_decay": hp.loguniform("weight_decay", np.log(5e-7), np.log(2e-6)),
    "window_size": hp.choice("window_size", [64, 256]),
    "use_batch_norm": hp.choice("use_batch_norm", [False]),
    "minibatch_size": 64 + hp.randint("minibatch_size", 193),
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
        use_hsmd=False,
        model_file="hyperopt_risk_neutral_no_mixture.pt",
        existing_model=None,
        symbols=SYMBOLS,
        refresh=False,
        risk_neutral=True,
        mixture_components=1,
        only_embeddings=False,
        max_epochs=400,
        early_termination=20,
        **parameters,
    )[1]

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
