'''
Flask app for NIH grant analyzer
'''
# pylint: disable=R0914
import os
import time

from flask import Flask, request, jsonify
from gensim.models import LdaModel
from gensim.similarities import MatrixSimilarity
from gensim import corpora

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge

from src.text import preprocess
from src.transformers import PreprocessTokensTransformer

# constants
API_KEY = os.getenv('API_KEY', '_')
MIN_DF = 3
MAX_DF = 0.6

app = Flask(__name__) # pylint: disable=C0103

# load data
dictionary = corpora.Dictionary.load('data/lda.dictionary') # pylint: disable=C0103
lda = LdaModel.load('data/lda.model') # pylint: disable=C0103
similarities = MatrixSimilarity.load('data/lda.similarity') # pylint: disable=C0103
data = pd.read_csv('data/data.csv', index_col='index', # pylint: disable=C0103
                   usecols=['index', 'project_number', 'title', 'abstract', 'rcr', 'preprocessed_text'])
topic_counts = pd.read_csv('data/lda.topic_counts.csv', index_col='year') # pylint: disable=C0103
topic_counts.columns = topic_counts.columns.astype(int)

def get_most_similar_documents(query, k=10):
    """Return indices for similar documents"""
    stime = time.time()
    sims = -similarities[query]
    print('Calculating similarities took {:.2f} seconds'.format(time.time() - stime))
    return sims.argsort()[:k]

def get_similar_documents(tokens, k=10, n_topics=3):
    """Return the project number, title, abstract for the similar documents"""
    stime = time.time()
    topic_dist = lda[dictionary.doc2bow(tokens)]
    bow_matrix = np.array([tup[1] for tup in topic_dist])
    print('Calculating new BoW matrix took {:.2f} seconds'.format(time.time()-stime))

    indices = get_most_similar_documents(bow_matrix, k)
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
    cleaned_tokens = data.loc[indices, 'preprocessed_text'].astype(str).str.split().values
    # cleaned_tokens = [row.split() if isinstance(row,str) else [] for row in data.loc[indices, 'preprocessed_text']] # pylint: disable=E1101
    print('Tokenizing collected documents took {:.2f} seconds'.format(time.time()-stime))
    stime = time.time()
    X = preprocessor.fit_transform(cleaned_tokens) # pylint: disable=C0103
    print('Vectorizing preprocessed text took {:.2f} seconds'.format(time.time()-stime))

    stime = time.time()
    X_test = preprocessor.transform([tokenized_text]) # pylint: disable=C0103
    y = scores>0
    model = LogisticRegression(C=2.5).fit(X, y)
    print('Logistic regression took {:.2f} seconds'.format(time.time()-stime))

    stime = time.time()
    if model.predict(X_test)[0]:
        indices = np.where(scores)[0]
        # indices = [i for i in range(len(scores)) if scores[i] > 0]
        y = np.log10(scores[indices])
        # model = Ridge(alpha=5).fit(X[indices, :], y)
        # score_pred = 10**model.predict(X_test)[0]
        # coef = model.coef_
        corr = np.corrcoef(X[indices, :].toarray(), y, rowvar=False)[-1,:-1]
        _, pval = f_regression(X[indices, :], y)
    else:
        # score_pred = 0.
        # corr = model.coef_[0]
        corr = np.corrcoef(X.toarray(), y, rowvar=False)[-1,:-1]
        _, pval = f_classif(X, y)
    print('Calculating corr & pval took {:.2f} seconds'.format(time.time()-stime))

    stime = time.time()
    words = preprocessor.vectorizer.get_feature_names()
    word_indices = corr.argsort()

    topwords, botwords = [], []
    for i in word_indices[::-1]:
        if len(topwords) >= k or corr[i] <= 0.:
            break
        if pval[i] < 0.05:
            topwords.append((words[i], corr[i]))
    for i in word_indices:
        if len(botwords) >= k or corr[i] >= 0.:
            break
        if pval[i] < 0.05:
            botwords.append((words[i], corr[i]))
    # topwords = word_rankings[:-k-1:-1]
    # botwords = word_rankings[:k:1]
    # botwords = []
    # for word, coef in word_rankings:
    #     if coef >= 0:
    #         break
    #     if word in tokenized_text:
    #         botwords.append((word, coef))
    #     if len(botwords) >= k:
    #         break
    print('Collecting processed word recommendations took {:.2f} seconds'.format(time.time()-stime))
    return topwords, botwords


@app.route('/', methods=['POST'])
def index():
    """main model API"""
    stime = time.time()
    if request.args.get('api_key') != API_KEY:
        return 'API key missing or wrong', 403
    if 'abstract' in request.form:
        abstract = request.form['abstract']
    else:
        return 'Abstract missing from form', 403
    n_topics = request.args.get('n_topics', 3, int)
    n_words = request.args.get('n_words', 5, int)
    n_abstracts = request.args.get('n_abstracts', 10, int)
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

    scores = data.loc[ids, 'rcr'].values # pylint: disable=E1101
    topwords, botwords = return_topbot_words(ids, scores, tokenized_text, k=n_words)

    result = {'abstracts': documents[:n_abstracts],
              'suggestions': {'top': topwords, 'bot': botwords},
              'topics': topics_data}

    print('Processing entire request took {:.2f} seconds'.format(time.time()-stime))
    return jsonify(result)

@app.route('/topic', methods=['GET'])
def topic():
    """Returns topic information"""
    if request.args.get('api_key') != API_KEY:
        return 'API key missing or wrong', 403

    N_topics = similarities.index.shape[1]
    try:
        topic = int(request.args.get('topic'))
        n_words = int(request.args.get('n_words'))
        assert topic>=0 and topic<N_topics
        n_words = min(max(n_words, 10), 40)
    except (ValueError, TypeError):
        return 'Invalid data type. Topic number and n_words must be integers.', 400
    except AssertionError:
        return 'Topic number must be between 0 and {}'.format(N_topics-1), 400

    k_abstracts = 20
    indices = get_most_similar_documents([(topic, 1.) if i==topic else (i, 0.) for i in range(N_topics)],
                                         k=k_abstracts)
    documents = [tuple(data.loc[i, ['project_number', 'title', 'abstract']].values) # pylint: disable=E1101
                 for i in indices]

    topic_data = {'id': int(topic),
                  'years': list(map(int, topic_counts.index)),
                  'counts': list(map(int, topic_counts.loc[:, topic])),
                  'words': [t[0] for t in lda.show_topic(topic, n_words)],
                  'word_dist': [float(t[1]) for t in lda.show_topic(topic, n_words)],
                  'n_primary': int((similarities.index.argmax(axis=1)==topic).sum())}
    result = {'abstracts': documents,
              'topic': topic_data}
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # pylint: disable=C0103
    app.run(host='0.0.0.0', port=port)
