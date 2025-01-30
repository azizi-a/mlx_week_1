import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import tqdm
import wandb
from config import (
    EMBEDDING_DIM, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    NUM_NEGATIVE_SAMPLES, TOP_K_SIMILAR,
)

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

def train_model(corpus: list, words_to_ids: Dict[str, int], 
                ids_to_words: Dict[int, str],
                embedding_dim: int = EMBEDDING_DIM, 
                epochs: int = EPOCHS, 
                batch_size: int = BATCH_SIZE, 
                learning_rate: float = LEARNING_RATE):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(words_to_ids)
    
    # Initialize model
    model = SkipGram(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert corpus to tensor
    corpus_ids = [words_to_ids[word] for word in corpus]
    corpus_tensor = torch.LongTensor(corpus_ids)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm.tqdm(range(0, len(corpus_ids) - batch_size, batch_size))
        
        for i in progress_bar:
            batch_inputs = corpus_tensor[i:i+batch_size].to(device)
            batch_targets = corpus_tensor[i+1:i+1+batch_size].to(device)
            
            negatives = torch.randint(0, vocab_size, (batch_size, NUM_NEGATIVE_SAMPLES)).to(device)
            
            optimizer.zero_grad()
            loss = model(batch_inputs, batch_targets, negatives)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({'loss': loss.item()})
            progress_bar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        
        # Test model after each epoch using dot product similarity
        test_words = ['king', 'queen', 'man', 'woman', 'dog', 'cat']
        print(f"\nEpoch {epoch+1} similar words embedding:")
        for word in test_words:
            if word in words_to_ids:
                print(f"\nWords similar to '{word}':")
                similar_words = get_similar_words(model, word, words_to_ids, ids_to_words)
                for similar_word, similarity in similar_words:
                    print(f"{similar_word.replace('_', '-')}: {similarity:.3f}")
        print('==============================================\n')
    
    return model

def get_similar_words(model, word: str,
                     words_to_ids: Dict[str, int], 
                     ids_to_words: Dict[int, str], 
                     top_k: int = TOP_K_SIMILAR):
    """Find similar words using trained model."""
    device = next(model.parameters()).device
    model.eval()
    
    if word not in words_to_ids:
        print(f"'{word}' not found in vocabulary")
        return []

    with torch.no_grad():
        word_id = words_to_ids[word]
        word_tensor = torch.LongTensor([word_id]).to(device)
        word_embedding = model.embeddings(word_tensor)
        all_embeddings = model.forward_weight_matrix.weight
        
        similarities = torch.cosine_similarity(word_embedding, all_embeddings)
        top_values, top_indices = torch.topk(similarities, k=top_k)
        
        similar_words = []
        for idx, similarity in zip(top_indices, top_values):
            similar_word = ids_to_words[idx.item()]
            if similar_word != word:
                similar_words.append((similar_word, similarity.item()))
    
    return similar_words 
