# Third party packages
import torch

# Local packages
import models


class SingleSymbolModelFromEmbedding(torch.nn.Module):
    def __init__(self, network, embedding):
        super().__init__()
        self.network = network
        self.embedding = embedding

        # Client code reads the window_size attribute :(
        self.window_size = network.window_size

    def forward(self, context, return_latents=False):
        minibatch_dim = context.shape[0]
        embedding_dim = len(self.embedding)
        embedding = self.embedding.unsqueeze(0).expand(minibatch_dim, embedding_dim)

        return self.network.forward((context, embedding))

    def dimensions(self):
        return self.network.dimensions()


def SingleSymbolModelFactory(encoding, model_with_embedding):
    model = model_with_embedding.network.model
    embeddings = model_with_embedding.network.embedding

    def single_symbol_model(symbol):
        this_embedding = embeddings(torch.tensor(encoding[symbol])).detach()
        return models.StockModel(
            symbols=(symbol.upper(),),
            network=SingleSymbolModelFromEmbedding(model, this_embedding),
            date=model_with_embedding.date,
            epochs=model_with_embedding.epochs,
            loss=model_with_embedding.loss,
        )

    return single_symbol_model
