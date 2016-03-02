from flask import Flask, render_template, request, Response, jsonify
from collections import Counter

from candidate_classifier.build_models import get_models


app = Flask(__name__)

MODELS = get_models()


@app.route('/')
def index():
    # TODO: Maybe don't bother with any templating?
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    # FIXME: Encodings?
    # Get the text
    text = request.get_data()
    # TODO: Process text
    text = text.lower().split()

    # Get probability estimates
    # NB: These are log probabilities
    probs = Counter({name: MODELS[name].prob_seq(text) for name in MODELS.keys()})

    # Get most likely
    result = probs.most_common(1)[0][0]

    return result






if __name__ == '__main__':
    app.run(debug=True)
