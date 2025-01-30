"""Upvote prediction model using Word2Vec embeddings."""

import torch
import torch.nn as nn
import config

class UpvotePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        layers = []
        input_dim = config.EMBEDDING_DIM
        
        # Build hidden layers
        for hidden_dim in config.HIDDEN_LAYERS:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model 