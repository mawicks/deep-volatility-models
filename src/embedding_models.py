from typing import Callable, Dict

# Third party packages
import torch

# Local packages
import model_wrappers


class SingleSymbolModelFromEmbedding(torch.nn.Module):
    def __init__(self, network: torch.nn.Module, single_embedding: torch.Tensor):
        super().__init__()
        self.network = network
        self.single_embedding = single_embedding

        # Client code reads the window_size attribute :(
        self.window_size = network.window_size

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        minibatch_dim = window.shape[0]
        embedding_dim = len(self.single_embedding)
        embedding = self.single_embedding.unsqueeze(0).expand(
            minibatch_dim, embedding_dim
        )

        return self.network.forward((window, embedding))


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
        )

    return single_symbol_model
