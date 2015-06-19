from nltk import SnowballStemmer
from nltk import clean_html
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

# Tokenizing (Document to list of sentences. Sentence to list of words.)
def tokenize(txt):
    '''Tokenize (sentence then words) filter out punctuation and change to lower case.'''
    tokens = []
    sentences = sent_tokenize(txt.replace("'", ""))
    sentences = [" ".join(re.findall(r'\w+', s, flags = re.UNICODE | re.LOCALE)).lower() for s in sentences]
    for stn in sentences:
        tokens += word_tokenize(stn)
    return tokens


# The preprocess pipeline. Returns as lists of tokens or as string.
# If stemmer_type = False or not supported then no stemming.
def tokenize_and_stem(txt, stem=True, remove_html=True, join=False, remove_stopwords=True):
    ''' Remove html and stopwords, tokenize and stem. '''

    lang = 'english'
    if remove_html:
        txt = clean_html(txt)

    words = tokenize(txt)
    if remove_stopwords:
        stop_words = stopwords.words(lang)
        words = [w for w in words if w.lower() not in stop_words]

    if stem:
        stemmer = SnowballStemmer(lang)
        words = [stemmer.stem(word).encode(encoding="utf8") for word in words]

    if join:
        words = " ".join(words)

    return words

