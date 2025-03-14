import collections
import typing
import word2vec.config as config
import nltk
import json

# Download required NLTK data
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


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


def lemmatize_text(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())

    # Lemmatize each token with its POS tag
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens
    ]

    return lemmatized_tokens


def preprocess(
    text: str, min_word_frequency: int = config.MIN_WORD_FREQUENCY
) -> typing.List[str]:
    """Preprocess text and return list of words."""
    words = lemmatize_text(text)
    # Add special tokens for punctuation
    special_tokens = [
        (".", "<PERIOD>"),
        (",", "<COMMA>"),
        ('"', "<QUOTATION_MARK>"),
        (";", "<SEMICOLON>"),
        ("!", "<EXCLAMATION_MARK>"),
        ("?", "<QUESTION_MARK>"),
        ("(", "<LEFT_PAREN>"),
        (")", "<RIGHT_PAREN>"),
        ("-", "<HYPHEN>"),
        (":", "<COLON>"),
        ("[", "<LEFT_BRACKET>"),
        ("]", "<RIGHT_BRACKET>"),
        ("{", "<LEFT_BRACE>"),
        ("}", "<RIGHT_BRACE>"),
        ("/", "<SLASH>"),
        ("\\", "<BACKSLASH>"),
        ("*", "<ASTERISK>"),
        ("&", "<AMPERSAND>"),
        ("#", "<HASH>"),
        ("@", "<AT>"),
        ("_", "<UNDERSCORE>"),
        ("=", "<EQUALS>"),
        ("+", "<PLUS>"),
        ("-", "<MINUS>"),
        ("%", "<PERCENT>"),
        ("^", "<CARET>"),
        ("~", "<TILDE>"),
        ("|", "<PIPE>"),
        ("”", "<QUOTATION_MARK>"),
        ("“", "<QUOTATION_MARK>"),
    ]

    for orig, replacement in special_tokens:
        for word in text:
            if word == orig:
                word = replacement

    stats = collections.Counter(words)

    # Remove the most frequent words
    most_common_words = [
        word for word, _ in stats.most_common(config.TOP_K_WORDS_TO_REMOVE)
    ]
    print("Common removed words corpus:", most_common_words)
    print("Unique words in corpus:", len(words))

    return [
        word
        for word in words
        if stats[word] > min_word_frequency and word not in most_common_words
    ]


def create_lookup_tables(
    words: typing.List[str],
) -> typing.Tuple[typing.Dict[str, int], typing.Dict[int, str]]:
    """Create lookup tables for words to indices and vice versa."""
    word_counts = collections.Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    words_to_ids = {word: i for i, word in enumerate(sorted_vocab)}
    ids_to_words = {i: word for word, i in words_to_ids.items()}

    # Save lookup tables to files
    with open("model/word_to_id.json", "w") as f:
        json.dump(words_to_ids, f)

    with open("model/id_to_word.json", "w") as f:
        json.dump(ids_to_words, f)

    print(f"Saved lookup tables with {len(words_to_ids)} words")

    return words_to_ids, ids_to_words
