"""Configuration file for Hacker News upvote prediction model."""

# Database configs
DB_URL = 'postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'

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
VOCAB_SIZE =  17138 #25155616
EMBEDDING_DIM = 128
HIDDEN_LAYERS = [32, 16, 8] #[256, 128, 64]  # Size of hidden layers
DROPOUT_RATE = 0.2

# Training configs
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 1

# Model saving/loading
MODEL_SAVE_PATH = 'upvote_predictor.pt'

# Evaluation metrics
METRICS = ['mae', 'mse', 'rmse', 'r2'] 