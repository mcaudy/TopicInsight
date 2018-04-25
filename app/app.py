'''
Flask app for NIH grant analyzer
'''
# pylint: disable=R0914
import os
import re
import time

from flask import Flask, request, jsonify
from gensim.models import LdaModel
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.stats import entropy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sqlalchemy import create_engine

from src.transformers import PreprocessTokensTransformer

# constants
API_KEY = os.getenv('API_KEY', '_')
PSQL_PATH = 'postgresql+psycopg2://{}:{}@{}/{}'.format(os.environ['DB_USER'],
                                                       os.environ['DB_PW'],
                                                       os.environ['DB_URI'],
                                                       os.environ['DB_NAME'])
MIN_DF = os.getenv('COUNTVECTORIZER_MIN_DF', 7)
MAX_DF = os.getenv('COUNTVECTORIZER_MAX_DF', 0.6)

app = Flask(__name__) # pylint: disable=C0103

# load data
dictionary = corpora.Dictionary.load('data/lda.dictionary') # pylint: disable=C0103
lda = LdaModel.load('data/lda.model') # pylint: disable=C0103
matrix = np.load('data/lda.matrix.npy') # pylint: disable=C0103

engine = create_engine(PSQL_PATH, echo=False) # pylint: disable=C0103
conn = engine.connect() # pylint: disable=C0103

q = ('SELECT a.index, a.project_number, b.title, b.abstract, b.rcr, c.preprocessed_text ' # pylint: disable=C0103
     'FROM index2projectnumber a '
     'LEFT JOIN data b '
     'ON a.project_number = b.project_number '
     'LEFT JOIN data_clean c '
     'ON a.project_number = c.project_number;')
data = pd.read_sql_query(q, conn, 'index') # pylint: disable=C0103

conn.close()

topic_counts = pd.read_csv('data/topic_counts_per_year.csv', index_col='year') # pylint: disable=C0103
topic_counts.columns = topic_counts.columns.astype(int)

def initial_clean(text):
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

stop_words = stopwords.words('english') # pylint: disable=C0103
stop_words += ['applic', 'descript', 'provid', 'unread', 'project', 'propos', 'abstract', # pylint: disable=C0103
               'goal', 'summari', 'research', 'object', 'term', 'studi', 'investig']
stemmer = PorterStemmer() # pylint: disable=C0103
def remove_stopwords_and_stem_words(text):
    """Function to stem words, so plural and singular are treated the same"""
    return [stemmer.stem(word) for word in text if word not in stop_words]

def preprocess(text):
    """Combine initial clean + remove/stem"""
    return remove_stopwords_and_stem_words(initial_clean(text))

def jensen_shannon(mat1, mat2):
    """similarity measure for two topics"""
    p_ = mat1[None, :].T # pylint: disable=C0103
    q_ = mat2.T # pylint: disable=C0103
    m = 0.5 * (p_ + q_) # pylint: disable=C0103
    return np.sqrt(entropy(p_, m) + entropy(q_, m)) # technically 0.5* inside sqrt

def get_most_similar_documents(query, document_matrix, k=10):
    """Return indices for similar documents"""
    stime = time.time()
    # sims = jensen_shannon(query, document_matrix)
    # print('Calculating Jensen-Shannon distance took {:.2f} seconds'.format(time.time()-stime))
    sims = -np.dot(document_matrix, query)
    print('Calculating dot product took {:.2f} seconds'.format(time.time()-stime))
    return sims.argsort()[:k]

def get_similar_documents(tokens, k=10, n_topics=3):
    """Return the project number, title, abstract for the similar documents"""
    stime = time.time()
    topic_dist = lda.get_document_topics(bow=dictionary.doc2bow(tokens))
    bow_matrix = np.array([tup[1] for tup in topic_dist])
    print('Calculating new BoW matrix took {:.2f} seconds'.format(time.time()-stime))

    indices = get_most_similar_documents(bow_matrix, matrix, k)
    stime = time.time()
    documents = [tuple(data.loc[i, ['project_number', 'title', 'abstract']].values) # pylint: disable=E1101
                 for i in indices]
    print('Extracting documents from dataframe took {:.2f} seconds'.format(time.time()-stime))

    stime = time.time()
    topic_indices = bow_matrix.argsort()[:-n_topics-1:-1]
    topic_matrix = [bow_matrix[i] for i in topic_indices]
    print('Collecting processed documents took {:.2f} seconds'.format(time.time()-stime))

    return list(indices), documents, list(topic_indices), topic_matrix

def return_topbot_words(indices, scores, tokenized_text, k=5):
    """Return words that contribute to high or low score abstracts"""
    preprocessor = PreprocessTokensTransformer(CountVectorizer(min_df=MIN_DF,
                                                               max_df=MAX_DF),
                                               TfidfTransformer())
    stime = time.time()
    # cleaned_texts = [preprocess(doc[2]) for doc in documents]
    cleaned_tokens = [row.split() if row else [] for row in data.loc[indices, 'preprocessed_text']] # pylint: disable=E1101
    print('Tokenizing collected documents took {:.2f} seconds'.format(time.time()-stime))
    stime = time.time()
    X = preprocessor.fit_transform(cleaned_tokens) # pylint: disable=C0103
    print('Vectorizing preprocessed text took {:.2f} seconds'.format(time.time()-stime))

    stime = time.time()
    X_test = preprocessor.transform([tokenized_text]) # pylint: disable=C0103
    model = LogisticRegression(C=2.5).fit(X, [int(s > 0) for s in scores])
    print('Logistic regression took {:.2f} seconds'.format(time.time()-stime))

    if model.predict(X_test)[0]:
        stime = time.time()
        indices = [i for i in range(len(scores)) if scores[i] > 0]
        model = Ridge(alpha=1.5).fit(X[indices, :], [np.log10(scores[i]) for i in indices])
        score_pred = 10**model.predict(X_test)[0]
        print('Linear regression took {:.2f} seconds'.format(time.time()-stime))
    else:
        score_pred = 0.

    stime = time.time()
    words = preprocessor.vectorizer.get_feature_names()
    # word_rankings is list of (word, coefficient value) from low to high coef
    word_rankings = [(words[i], model.coef_[i]) for i in model.coef_.argsort()]

    topwords = word_rankings[:-k-1:-1]
    botwords = []
    for word, coef in word_rankings:
        if coef >= 0:
            break
        if word in tokenized_text:
            botwords.append((word, coef))
        if len(botwords) >= k:
            break
    print('Collecting processed word recommendations took {:.2f} seconds'.format(time.time()-stime))
    return topwords, botwords, score_pred


@app.route('/', methods=['POST'])
def index():
    """main model API"""
    stime = time.time()
    if request.args.get('api_key') != API_KEY:
        return 'API key missing or wrong', 403
    if ('abstract' in request.form) and ('n_topics' in request.form):
        abstract = request.form['abstract']
        n_topics = int(request.form['n_topics'])
    else:
        return 'Abstract missing from form', 403
    tokenized_text = preprocess(abstract)

    ids, documents, topic_indices, topic_dist = get_similar_documents(tokenized_text,
                                                                      k=1000,
                                                                      n_topics=n_topics)

    words = [[t[0] for t in lda.show_topic(d)] for d in topic_indices]

    topics_data = [{'id': int(tid),
                    'years': list(map(int, topic_counts.index)),
                    'counts': list(map(int, topic_counts.loc[:, tid])),
                    'dist': float(topic_dist[i]),
                    'words': words[i]} for i, tid in enumerate(topic_indices)]

    scores = list(data.loc[ids, 'rcr']) # pylint: disable=E1101
    topwords, botwords, pred = return_topbot_words(ids, scores, tokenized_text, k=5)

    result = {'abstracts': documents[:10],
              'suggestions': {'top': topwords, 'bot': botwords},
              'topics': topics_data,
              'predicted_score': pred}

    print('Processing entire request took {:.2f} seconds'.format(time.time()-stime))
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # pylint: disable=C0103
    app.run(host='0.0.0.0', port=port, debug=True)
