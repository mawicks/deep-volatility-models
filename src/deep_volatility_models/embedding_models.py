from typing import Callable, Dict

# Third party packages
import torch

# Local packages
from deep_volatility_models import model_wrappers
from deep_volatility_models import sample


class SingleSymbolModelFromEmbedding(torch.nn.Module):
    def __init__(self, network: torch.nn.Module, single_embedding: torch.Tensor):
        super().__init__()
        self.network = network
        self.single_embedding = single_embedding

        # Client code reads the window_size attribute :(
        self.window_size = network.window_size

    @property
    def is_mixture(self):
        return self.network.is_mixture

    def make_predictors(self, window: torch.Tensor) -> torch.Tensor:
        """
        Combine the `window` and the `embedding` to make `predictors` input for
        use with the underlying network.
        """

        minibatch_dim = window.shape[0]
        embedding_dim = len(self.single_embedding)
        embedding = self.single_embedding.unsqueeze(0).expand(
            minibatch_dim, embedding_dim
        )
        predictors = (window, embedding)
        return predictors

    def simulate_one(
        self,
        window: torch.Tensor,
        time_samples: int,
    ):
        return sample.simulate_one(
            self.network,
            self.make_predictors(window),
            time_samples,
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.network.forward(self.make_predictors(window))


def SingleSymbolModelFactory(
    encoding: Dict[str, int], wrapped_model: model_wrappers.StockModel
) -> Callable[[str], model_wrappers.StockModel]:
    if isinstance(wrapped_model.network.model, torch.nn.Module):
        model = wrapped_model.network.model
    else:
        raise ValueError(
            "wrapped_model must have `network` field with `model` of type `Module`"
        )

    if isinstance(wrapped_model.network.embedding, torch.nn.Module):
        embeddings = wrapped_model.network.embedding
    else:
        raise ValueError(
            "wrapped_model must have `network` field with `embeddings` of type `Module`"
        )

    def single_symbol_model(symbol: str) -> model_wrappers.StockModel:
        single_embedding = embeddings(torch.tensor(encoding[symbol])).detach()
        return model_wrappers.StockModel(
            symbols=(symbol.upper(),),
            network=SingleSymbolModelFromEmbedding(model, single_embedding),
            date=wrapped_model.date,
            epochs=wrapped_model.epochs,
            loss=wrapped_model.loss,
            training_data_start_date=None,
            training_data_end_date=None,
        )

    return single_symbol_model
