import logging

from hyperopt import hp, tpe, fmin, Trials
import numpy as np

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

search_space = {
    "mixture_components": 3 + hp.randint("mixture_components", 4),
    "feature_dimension": 40 + hp.randint("feature_dimension", 51),
    "embedding_dimension": 3 + hp.randint("embedding_dimension", 13),
    "gaussian_noise": hp.loguniform("gaussian_noise", np.log(1e-4), np.log(1e-2)),
    "dropout": hp.uniform("dropout", 0, 0.25),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-8), np.log(1e-6)),
    "window_size": hp.choice("window_size", [64, 256]),
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
