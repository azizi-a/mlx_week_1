"""Configuration file for Word2Vec implementation."""

# Data processing configs
DATA_URL = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8"
DATA1_PATH = "text8"
WORDS_TO_PROCESS_COUNT = 100_000#_000 # Number of words to use for training
MIN_WORD_FREQUENCY = 8  # Minimum frequency for a word to be included in vocabulary
TOP_K_WORDS_TO_REMOVE = 40 # Number of most frequent words to remove from vocabulary

# Hacker news data
DB_URL = 'postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'
DATA2_PATH = "hn_data"
QUERY = """
    SELECT title FROM hacker_news.items
    WHERE "type" = 'story'
    AND title IS NOT NULL
    LIMIT 1_000_000;
    """

# Model architecture configs
EMBEDDING_DIM = 128  # Dimension of word embeddings
CONTEXT_WINDOW = 2  # Words on each side to consider as context

# Training configs
BATCH_SIZE = 2048
LEARNING_RATE = 0.001
EPOCHS = 8
NUM_NEGATIVE_SAMPLES = 2  # Number of negative samples per positive sample

# Model saving/loading
MODEL_SAVE_PATH = 'weights.pt'

# Testing configs
TOP_K_SIMILAR = 5  # Number of similar words to return when testing

# Device configuration
USE_CUDA = True  # Will fall back to CPU if CUDA is not available

