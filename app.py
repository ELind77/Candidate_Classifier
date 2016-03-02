from flask import Flask, render_template, request, Response, jsonify
from collections import Counter
import json
from spacy.en import English
import string

from candidate_classifier.build_models import get_models
from candidate_classifier.string_processing import TransformerABC



app = Flask(__name__)

MODELS = get_models()


#
# String Processing
# Functions for processing string input from users
# Just does basic cleaning and tokenization

NLP = English(entity=False, tagger=False, parser=False, load_vectors=False)
PUNCT = frozenset(string.punctuation)
def add_punct(s):
    if s[-1] not in PUNCT:
         s += u'.'
    return s

def tokenizer(s):
    return ['<S>'] + [t.lower_ for t in NLP(s)] + ['</S>']

STRING_PROCESSOR = TransformerABC(prefilter_substitutions=['html',
                                                           'whitespace',
                                                           'strip',
                                                           'deaccent',
                                                           add_punct],
                                  tokenizer=tokenizer)

#
# ROUTES
#

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
    processed = STRING_PROCESSOR(text)

    # Get probability estimates
    # NB: These are negative log probabilities, so to get the
    # Counter methods to work the values need to be negative.
    probs = Counter({name: -MODELS[name].prob_seq(processed) for name in MODELS.keys()})

    # Get most likely
    result = probs.most_common(1)[0][0]

    print processed
    print result
    print json.dumps(probs, indent=2)

    return result






if __name__ == '__main__':
    app.run(debug=True)
