import wandb

from get_data import download_text8, get_data_from_postgres
from preprocess import preprocess, create_lookup_tables
from SkipGram import train_model, get_similar_words
import torch
from config import (
    WORDS_TO_PROCESS_COUNT,
    MODEL_SAVE_PATH,
    QUERY,
    DATA2_PATH
)

if __name__ == "__main__":
    # Get text8 data
    text8_data = download_text8()
    
    # Get Hacker News data
    hn_data = get_data_from_postgres(QUERY, DATA2_PATH)
    
    # Combine both text sources
    combined_text = text8_data[:WORDS_TO_PROCESS_COUNT] + " " + hn_data[:WORDS_TO_PROCESS_COUNT]
    
    # Preprocess combined data
    corpus = preprocess(combined_text)
    words_to_ids, ids_to_words = create_lookup_tables(corpus)
    
    # Initialize wandb
    wandb.init(
        project='word2vec',
        config={
            'dataset': 'text8 + hn_data',
            'vocab_size': len(words_to_ids),
        }
    )

    # Train model
    model = train_model(corpus, words_to_ids, ids_to_words)

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Test model
    test_words = ['king', 'queen', 'man', 'woman', 'dog', 'cat']
    for word in test_words:
        if word in words_to_ids:
            print(f"\nWords similar to '{word}':")
            similar_words = get_similar_words(model, word, words_to_ids, ids_to_words)
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.3f}") 

    # Save model to wandb
    print('Saving...')
    torch.save(model.state_dict(), './weights.pt')
    print('Uploading...')
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file('./weights.pt')
    wandb.log_artifact(artifact)
    print('Done!')
    wandb.finish()

