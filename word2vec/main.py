import wandb
import torch

from word2vec import config, data, preprocessing, model as m


def main():
    # Get text8 data
    text8_data = data.download_text8()

    # Get Hacker News data
    hn_data = data.get_data_from_postgres(config.QUERY, config.DATA2_PATH)

    # Combine both text sources
    combined_text = (
        text8_data[: config.WORDS_TO_PROCESS_COUNT]
        + " "
        + hn_data[: config.WORDS_TO_PROCESS_COUNT]
    )

    print("Combined text length:", len(combined_text))

    # Preprocess combined data
    corpus = preprocessing.preprocess(combined_text)
    words_to_ids, ids_to_words = preprocessing.create_lookup_tables(corpus)

    # Initialize wandb
    wandb.init(
        project="word2vec",
        config={
            "dataset": "text8 + hn_data",
            "vocab_size": len(words_to_ids),
        },
    )

    # Train model
    model = m.train_model(corpus, words_to_ids, ids_to_words)

    # Save model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    # Test model
    test_words = ["king", "queen", "man", "woman", "dog", "cat"]
    for word in test_words:
        if word in words_to_ids:
            print(f"\nWords similar to '{word}':")
            similar_words = m.get_similar_words(
                model, word, words_to_ids, ids_to_words
            )
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.3f}")

    # Save model to wandb
    print("Saving...")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("Uploading...")
    artifact = wandb.Artifact("model-weights", type="model")
    artifact.add_file(config.MODEL_SAVE_PATH)
    wandb.log_artifact(artifact)
    print("Done!")
    wandb.finish()


if __name__ == "__main__":
    main()
