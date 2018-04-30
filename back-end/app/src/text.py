import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = stopwords.words('english') # pylint: disable=C0103
stop_words += ['application', 'applications', 'apply', 'applied', 'applies', 'applicant', 'applicants',
               'description', 'descriptions', 'provide', 'provides', 'provided', 'providing', 'unread',
               'project', 'projects', 'propose', 'proposed', 'proposes', 'abstract', 'abstracts',
               'goal', 'goals', 'summary', 'research', 'researches', 'researched',
               'object', 'objective', 'objectives', 'term', 'terms', 'study', 'studies', 'studied', 'studying'
               'investigate', 'investigated']
stemmer = PorterStemmer() # pylint: disable=C0103
stop_words = set(stop_words)
stemmed_stop_words = set(stemmer.stem(word) for word in stop_words)

def _initial_clean(text):
    """
    Function to clean text of HTML tags, websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub(r"<[^>]{1,20}>", " ", text)
    text = re.sub(r"((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

def _remove_stopwords_and_stem_words(text):
    """Function to stem words, so plural and singular are treated the same"""
    return [stemmer.stem(word) for word in text if word not in stop_words]

def preprocess(text):
    """Combine initial clean + remove/stem"""
    return [w for w in _remove_stopwords_and_stem_words(_initial_clean(text))
            if w not in stemmed_stop_words]