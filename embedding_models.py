# Third party packages
import torch

# Local packages
import model_wrappers


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


def SingleSymbolModelFactory(encoding, wrapped_model):
    model = wrapped_model.network.model
    embeddings = wrapped_model.network.embedding

    def single_symbol_model(symbol):
        this_embedding = embeddings(torch.tensor(encoding[symbol])).detach()
        return model_wrappers.StockModel(
            symbols=(symbol.upper(),),
            network=SingleSymbolModelFromEmbedding(model, this_embedding),
            date=wrapped_model.date,
            epochs=wrapped_model.epochs,
            loss=wrapped_model.loss,
        )

    return single_symbol_model
