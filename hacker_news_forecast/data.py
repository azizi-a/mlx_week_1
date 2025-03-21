import torch
import sklearn
import json
import nltk
import pandas as pd
import sqlalchemy

from hacker_news_forecast import config
import word2vec.model

# Download required NLTK data
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


class HNDataset(torch.utils.data.Dataset):
    def __init__(self, titles, scores):
        self.titles = titles
        self.scores = scores

        with open(config.WORD_TO_ID, "r") as f:
            self.word_to_id = json.load(f)

        # Load pretrained Word2Vec model
        self.word2vec = word2vec.model.SkipGram(
            len(self.word_to_id), config.EMBEDDING_DIM
        )
        self.word2vec.load_state_dict(torch.load(config.WORD2VEC_WEIGHTS))
        # Set to evaluation mode
        self.word2vec.eval()

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        score = self.scores[idx]

        # Lemmatize title
        lemmatized_words = lemmatize_text(title)

        # Convert lemmatized tokens to embeddings
        word_embeddings = []

        with open(config.WORD_TO_ID, "r") as f:
            word_to_id = json.load(f)

        for word in lemmatized_words:
            if word in word_to_id:
                embedding = self.word2vec.get_embedding(word_to_id[word])
                word_embeddings.append(embedding)

        if word_embeddings:
            # Average the word embeddings
            title_embedding = torch.stack(word_embeddings).mean(dim=0)
        else:
            # If no words found in vocabulary, use zero vector
            title_embedding = torch.zeros(config.EMBEDDING_DIM)

        return title_embedding, torch.tensor(score, dtype=torch.float32)


def gather_data():
    """Fetch story titles and scores from Hacker News database."""
    engine = sqlalchemy.create_engine(config.DB_URL)

    try:
        df = pd.read_sql(config.QUERY, engine)
        print(f"Retrieved {len(df)} records")
        df.to_csv(config.DATA_PATH, index=False)
        return df
    except Exception as e:
        print(f"Error gathering data: {e}")
        return None
    finally:
        engine.dispose()


def prepare_data(df):
    """Prepare data splits and create DataLoaders."""

    # Create train/val/test splits
    train_df, temp_df = sklearn.model_selection.train_test_split(
        df, train_size=config.TRAIN_SPLIT, random_state=config.RANDOM_SEED
    )

    val_df, test_df = sklearn.model_selection.train_test_split(
        temp_df,
        train_size=config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=config.RANDOM_SEED,
    )

    # Create datasets
    train_dataset = HNDataset(train_df["title"].values, train_df["score"].values)
    val_dataset = HNDataset(val_df["title"].values, val_df["score"].values)
    test_dataset = HNDataset(test_df["title"].values, test_df["score"].values)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
    )

    return train_loader, val_loader, test_loader


def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": nltk.corpus.wordnet.ADJ,
        "N": nltk.corpus.wordnet.NOUN,
        "V": nltk.corpus.wordnet.VERB,
        "R": nltk.corpus.wordnet.ADV,
    }
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def lemmatize_text(text: str):
    """Lemmatize text using NLTK."""
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # Handle case where text is not a string
    if not isinstance(text, str):
        text = str(text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens
    ]
    return lemmatized_tokens
