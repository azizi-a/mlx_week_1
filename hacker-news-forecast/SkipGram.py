import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Forward weight matrix - maps embeddings back to vocabulary size for prediction
        self.forward_weight_matrix = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, x, targets, negatives):
        emb = self.embeddings(x)
        pos_loss = torch.log(torch.sigmoid(torch.sum(emb * self.forward_weight_matrix.weight[targets], dim=1)))
        neg_loss = torch.log(torch.sigmoid(-torch.sum(emb.unsqueeze(1) * self.forward_weight_matrix.weight[negatives], dim=2)))
        return -(pos_loss + neg_loss.sum(1)).mean()
