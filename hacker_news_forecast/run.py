"""Training script for Hacker News upvote predictor."""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_gathering import gather_data
from data_processing import prepare_data
from model import UpvotePredictor
import config
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer):
    """Train the model and validate after each epoch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        # Add progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]')
        for titles, scores in train_pbar:
            titles, scores = titles.to(device), scores.to(device)
            
            optimizer.zero_grad()
            predictions = model(titles)
            loss = criterion(predictions, scores)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_losses = []
        
        # Add progress bar for validation
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Val]')
            for titles, scores in val_pbar:
                titles, scores = titles.to(device), scores.to(device)
                predictions = model(titles)
                loss = criterion(predictions, scores)
                val_losses.append(loss.item())
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(config.MODEL_SAVE_PATH)

def evaluate_model(model, test_loader):
    """Evaluate the model on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for titles, scores in test_loader:
            titles, scores = titles.to(device), scores.to(device)
            predictions = model(titles)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    
    print("\nTest Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

def main():
    # Gather data
    if not os.path.exists('hn_data.csv'):
        print("HN data not found. Running data_gathering.py first...")
        gather_data()
    df = pd.read_csv('hn_data.csv')
    if df is None:
        return
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(df)
    
    # Initialize model, criterion, and optimizer
    model = UpvotePredictor()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Initialize wandb
    wandb.init(
        project='word2vec',
        config={
            'dataset': 'hn_data',
            'model': 'UpvotePredictor',
        }
    )

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Load best model and evaluate
    best_model = UpvotePredictor.load(config.MODEL_SAVE_PATH)
    evaluate_model(best_model, test_loader)

    # Save model to wandb
    print('Uploading...')
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(config.MODEL_SAVE_PATH)
    wandb.log_artifact(artifact)
    print('Done!')
    wandb.finish()

if __name__ == "__main__":
    main() 