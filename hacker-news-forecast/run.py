"""Training script for Hacker News upvote predictor."""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_gathering import gather_data
from data_processing import prepare_data
from model import UpvotePredictor
import config

def train_model(model, train_loader, val_loader, criterion, optimizer):
    """Train the model and validate after each epoch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        for titles, scores in train_loader:
            titles, scores = titles.to(device), scores.to(device)
            
            optimizer.zero_grad()
            predictions = model(titles)
            loss = criterion(predictions, scores)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for titles, scores in val_loader:
                titles, scores = titles.to(device), scores.to(device)
                predictions = model(titles)
                loss = criterion(predictions, scores)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
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
    df = gather_data()
    if df is None:
        return
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(df)
    
    # Initialize model, criterion, and optimizer
    model = UpvotePredictor()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Load best model and evaluate
    best_model = UpvotePredictor.load(config.MODEL_SAVE_PATH)
    evaluate_model(best_model, test_loader)

if __name__ == "__main__":
    main() 