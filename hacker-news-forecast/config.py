"""Configuration file for Hacker News upvote prediction model."""

# Database configs
DB_URL = 'postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'

# Data gathering configs
QUERY = """
    SELECT title, score 
    FROM hacker_news.items
    WHERE "type" = 'story'
    AND title IS NOT NULL
    AND score IS NOT NULL
    LIMIT 1_000_000;
"""

# Data processing configs
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Model architecture configs
WORD2VEC_WEIGHTS = '../word2vec/weights.pt'
HIDDEN_LAYERS = [256, 128, 64]  # Size of hidden layers
DROPOUT_RATE = 0.2

# Training configs
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# Model saving/loading
MODEL_SAVE_PATH = 'upvote_predictor.pt'

# Evaluation metrics
METRICS = ['mae', 'mse', 'rmse', 'r2'] 