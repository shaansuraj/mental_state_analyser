import re
import emoji
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def clean_caption(caption):
    """
    Cleans the Instagram caption text by:
    - Removing hashtags, mentions, and URLs.
    - Removing emojis.
    - Removing punctuation (thus full stops, commas, etc.) and numbers.
    - Removing new line characters.
    - Lowercasing the text and trimming extra spaces.
    - Tokenizing and lemmatizing using spaCy.
    """
    # Remove hashtags, mentions, and URLs
    caption = re.sub(r'#\S+|@\S+|http\S+', '', caption)
    # Remove emojis
    caption = emoji.replace_emoji(caption, replace="")
    # Remove punctuation and numbers
    caption = re.sub(r'[^\w\s]', '', caption)
    caption = re.sub(r'\d+', '', caption)
    # Remove new line characters
    caption = caption.replace('\n', ' ')
    # Lowercase and remove extra spaces
    caption = caption.lower()
    caption = " ".join(caption.split())
    # Tokenize and lemmatize using spaCy
    doc = nlp(caption)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def preprocess_captions(captions):
    """
    Applies clean_caption to each caption in the list.
    """
    return [clean_caption(caption) for caption in captions]
