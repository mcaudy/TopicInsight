'''
Flask app for NIH grant analyzer
'''
import os
import urllib

import requests
from requests.exceptions import Timeout
from flask import Flask, render_template, request

import bokeh
from src.plots import trend_plot, histogram_plot, bar_plot, corr_plot

# constants
MODEL_API_URI = os.getenv('MODEL_API_URI', '')
API_KEY = os.getenv('API_KEY', '')

app = Flask(__name__) # pylint: disable=C0103

@app.route('/', methods=['GET'])
def index():
    """index page"""
    return render_template('index.html'), 200


@app.route('/lda', methods=['GET'])
def lda_page():
    """model explanation page"""
    return render_template('lda.html'), 200


@app.route('/nih', methods=['GET', 'POST'])
def test_case_page():
    """case page"""
    if request.method == 'GET':
        return render_template('nih.html', landing_page='true'), 200

    if 'abstract' in request.form:
        abstract = request.form['abstract']
    else:
        return render_template('nih.html',
                               error_msg='There was no abstract!'), 403

    try:
        r = requests.post(MODEL_API_URI, # pylint: disable=C0103
                          params={'api_key': API_KEY, 'n_topics': 3, 'n_words': 5, 'n_abstracts': 50},
                          data={'abstract': abstract},
                          timeout=60)
        data = r.json()
    except Timeout:
        return render_template('nih.html',
                               error_msg='Model API timed out... Try again another time.'), 500
    except: # pylint: disable=W0702
        return render_template('nih.html',
                               error_msg='There is an error with the model API...'), 500

    script, div = trend_plot(data['topics'], width=500, height=300)
    script2, div2 = histogram_plot(data['topics'], width=500, height=300)
    script3, div3 = corr_plot(data['suggestions'], width=600, height=450)

    return render_template('nih.html',
                           query=abstract,
                           abstracts=data['abstracts'],
                           topwords=data['suggestions']['top'],
                           botwords=data['suggestions']['bot'],
                           plot_div=div.strip(),
                           plot_script=script.strip(),
                           hist_div=div2.strip(),
                           hist_script=script2.strip(),
                           corr_div=div3.strip(),
                           corr_script=script3.strip(),
                           bokeh_version=bokeh.__version__), 200


@app.route('/topic/<int:topic>')
def topic(topic):
    try:
        r = requests.get(urllib.parse.urljoin(MODEL_API_URI, 'topic'),
                         params={'api_key': API_KEY,
                                 'topic': topic,
                                 'n_words': 30},
                         timeout=30)
        assert r.ok
    except Timeout:
        return render_template('topic.html',
                               error_msg='Our server seems to be down right now...'), 500
    except AssertionError:
        return render_template('topic.html',
                               error_msg='There is an issue with our server right now...'), 500
    except:
        return render_template('topic.html',
                               error_msg='There is an issue with our server right now...'), 500
    data = r.json()
    script, div = trend_plot([data['topic']], width=500, height=500)
    script2, div2 = bar_plot(data['topic'], width=500, height=500)
    
    return render_template('topic.html',
                           topic=topic,
                           n_primary=data['topic']['n_primary'],
                           abstracts=data['abstracts'],
                           plot_div=div.strip(),
                           plot_script=script.strip(),
                           bar_div=div2.strip(),
                           bar_script=script2.strip()), 200

@app.route('/grants/<project_number>')
def grant(project_number):
    return render_template('404.html',
                           error_msg='This page is under construction... Please come back later.'), 404

@app.route('/word_correlations', methods=['GET'])
def word_correlation_page():
    """word correlations page"""
    return render_template('word_correlations.html'), 200

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # pylint: disable=C0103
    app.run(host='0.0.0.0', port=port)
