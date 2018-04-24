'''
Flask app for NIH grant analyzer
'''
import os

import requests
from requests.exceptions import Timeout
from flask import Flask, render_template, request

from src.plots import trend_plot, histogram_plot

# constants
MODEL_API_URI = os.environ['MODEL_API_URI']
API_KEY = os.environ['API_KEY']

app = Flask(__name__) # pylint: disable=C0103

@app.route('/', methods=['GET', 'POST'])
def index():
    """index page"""
    if request.method == 'GET':
        return render_template('index.html', landing_page='true')

    if 'abstract' in request.form:
        abstract = request.form['abstract']
    else:
        return render_template('index.html',
                               error_msg='There was no abstract!')

    try:
        r = requests.post(MODEL_API_URI, # pylint: disable=C0103
                          params={'api_key': API_KEY},
                          data={'abstract': abstract, 'n_topics': 3},
                          timeout=60)
        data = r.json()
    except Timeout:
        return render_template('index.html',
                               error_msg='Model API timed out... Try again another time.')
    except: # pylint: disable=W0702
        return render_template('index.html',
                               error_msg='There is an error with the model API...')

    script, div = trend_plot(data['topics'], width=500, height=300)
    script2, div2 = histogram_plot(data['topics'], width=500, height=300)

    return render_template('index.html',
                           query=abstract,
                           abstracts=data['abstracts'],
                           topwords=data['suggestions']['top'],
                           botwords=data['suggestions']['bot'],
                           pred_score=data['predicted_score'],
                           plot_div=div.strip(),
                           plot_script=script.strip(),
                           hist_div=div2.strip(),
                           hist_script=script2.strip())

@app.route('/model', methods=['GET'])
def model_page():
    """model explanation page"""
    return render_template('model.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # pylint: disable=C0103
    app.run(host='0.0.0.0', port=port)
