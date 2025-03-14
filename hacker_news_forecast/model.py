import torch

from hacker_news_forecast import config


class UpvotePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        input_dim = config.EMBEDDING_DIM

        # Build hidden layers
        for hidden_dim in config.HIDDEN_LAYERS:
            layers.extend(
                [
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config.DROPOUT_RATE),
                ]
            )
            input_dim = hidden_dim

        # Output layer
        layers.append(torch.nn.Linear(input_dim, 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
