"""Data processing for Hacker News upvote prediction."""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from SkipGram import SkipGram  # Import SkipGram model
import config
import json
class HNDataset(Dataset):
    def __init__(self, titles, scores):
        self.titles = titles
        self.scores = scores

        with open('word_to_id.json', 'r') as f:
            self.word_to_id = json.load(f)
        
        # Load pretrained Word2Vec model
        self.word2vec = SkipGram.load(config.WORD2VEC_WEIGHTS, self.word_to_id, config.EMBEDDING_DIM)
        self.word2vec.eval()  # Set to evaluation mode
        
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        score = self.scores[idx]
        
        # Convert title to embedding
        words = title.lower().split()
        word_embeddings = []

        with open('word_to_id.json', 'r') as f:
            word_to_id = json.load(f)
        
        for word in words:
            if word in word_to_id:
                embedding = self.word2vec.get_embedding(word)
                word_embeddings.append(embedding)
        
        if word_embeddings:
            # Average the word embeddings
            title_embedding = torch.stack(word_embeddings).mean(dim=0)
        else:
            # If no words found in vocabulary, use zero vector
            title_embedding = torch.zeros(config.EMBEDDING_DIM)
            
        return title_embedding, torch.tensor(score, dtype=torch.float32)

def prepare_data(df):
    """Prepare data splits and create DataLoaders."""
    
    # Create train/val/test splits
    train_df, temp_df = train_test_split(
        df, train_size=config.TRAIN_SPLIT, random_state=config.RANDOM_SEED
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=config.VAL_SPLIT/(config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=config.RANDOM_SEED
    )
    
    # Create datasets
    train_dataset = HNDataset(train_df['title'].values, train_df['score'].values)
    val_dataset = HNDataset(val_df['title'].values, val_df['score'].values)
    test_dataset = HNDataset(test_df['title'].values, test_df['score'].values)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE
    )
    
    return train_loader, val_loader, test_loader 