import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, word_to_id: int, embedding_dim: int):
        super().__init__()
        self.word_to_id = word_to_id
        self.vocab_size = len(word_to_id)
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # Forward weight matrix - maps embeddings back to vocabulary size for prediction
        self.forward_weight_matrix = nn.Linear(embedding_dim, self.vocab_size, bias=False)
    def forward(self, x, targets, negatives):
        emb = self.embeddings(x)
        pos_loss = torch.log(torch.sigmoid(torch.sum(emb * self.forward_weight_matrix.weight[targets], dim=1)))
        neg_loss = torch.log(torch.sigmoid(-torch.sum(emb.unsqueeze(1) * self.forward_weight_matrix.weight[negatives], dim=2)))
        return -(pos_loss + neg_loss.sum(1)).mean()

    def load(path, vocab_size, embedding_dim):
        model = SkipGram(vocab_size, embedding_dim)
        model.load_state_dict(torch.load(path))
        return model
    
    def get_embedding(self, word):
        token = self.word_to_id[word]
        return self.embeddings(torch.tensor([token])).squeeze(0)
