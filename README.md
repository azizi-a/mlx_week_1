# Hacker News Forecast

## Overview

This project consists of two parts:

1. A Word2Vec model that generates embeddings for words in a corpus.
2. A simple model that uses the embeddings to predict the number of upvotes a
   story will receive on Hacker News.

## Word2Vec Model

The model is trained using the skip-gram objective on a corpus of Wikipedia and
Hacker News titles.

## Hacker News Forecast Model

The Hacker News Forecast model is a simple feedforward neural network that takes
in a story title and outputs a single number, which is the predicted number of
upvotes the story will receive.

## Installation

1. Install the dependencies using uv:

   ```bash
   uv sync
   ```

2. Activate the virtual environment using uv:

   ```bash
   uv venv
   ```

## Usage

1. Run the Word2Vec model
2. Run the Hacker News Forecast model

To do either of the above, run the following command and select the option you
want to run:

```bash
uv run main.py
```
