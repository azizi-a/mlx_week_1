import collections
from typing import List, Tuple, Dict
from config import MIN_WORD_FREQUENCY, TOP_K_WORDS_TO_REMOVE

def preprocess(text: str, min_word_frequency: int = MIN_WORD_FREQUENCY) -> List[str]:
    """Preprocess text and return list of words."""
    text = text.lower()
    # Add special tokens for punctuation
    special_tokens = [
        ('.', '<PERIOD>'), (',', '<COMMA>'), ('"', '<QUOTATION_MARK>'),
        (';', '<SEMICOLON>'), ('!', '<EXCLAMATION_MARK>'), ('?', '<QUESTION_MARK>'),
        ('(', '<LEFT_PAREN>'), (')', '<RIGHT_PAREN>'), ('-', '<HYPHEN>'),
        (':', '<COLON>'), ('[', '<LEFT_BRACKET>'), (']', '<RIGHT_BRACKET>'),
        ('{', '<LEFT_BRACE>'), ('}', '<RIGHT_BRACE>'), ('/', '<SLASH>'), ('\\', '<BACKSLASH>'),
        ('*', '<ASTERISK>'), ('&', '<AMPERSAND>'), ('#', '<HASH>'), ('@', '<AT>'), ('_', '<UNDERSCORE>'),
        ('=', '<EQUALS>'), ('+', '<PLUS>'), ('-', '<MINUS>'), ('%', '<PERCENT>'), ('^', '<CARET>'),
        ('~', '<TILDE>'), ('|', '<PIPE>'), 
    ]
    
    for orig, replacement in special_tokens:
        text = text.replace(orig, f' {replacement} ')
    
    words = text.split()
    stats = collections.Counter(words)
    
    # Remove the most frequent words
    most_common_words = [word for word, _ in stats.most_common(TOP_K_WORDS_TO_REMOVE)]
    print('Common removed words corpus:', most_common_words)
    print('Unique words in corpus:', len(words))

    return [word for word in words if stats[word] > min_word_frequency and word not in most_common_words]

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create lookup tables for words to indices and vice versa."""
    word_counts = collections.Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    
    words_to_ids = {word: i for i, word in enumerate(sorted_vocab)}
    ids_to_words = {i: word for word, i in words_to_ids.items()}
    
    return words_to_ids, ids_to_words 