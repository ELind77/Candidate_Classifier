Candidate Classifier
======

This is a small personal project to showcase some of what I've learned about Natural Language processing and web-app design.

The original inspiration for this project was a workshop paper by Grégoire Mesnil, Tomas Mikolov, Marc'Aurelio and Yoshua Bengio: [Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews](http://arxiv.org/abs/1412.5335).  The idea of using all of these different techniques together facinated me and I wanted to try it for myself.  The task I have chosen for my project is actually an authorship attribution task, the idea is to take a sentence, or short document and classify it as belonging to onw of the presidential hopefuls.
 
Currently, I'm using an ensemble of Multinomial Naive Bayes with a classifier based on an Ngram language model.


### Citations:
The debate transcriptions were taken from The American Presidency Project: `http://www.presidency.ucsb.edu`.

The NgramModel class is resurrected from the NLTK project and is mostly functional but still needs some work.

Grégoire Mesnil, Tomas Mikolov, Marc'Aurelio and Yoshua Bengio: Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews; Submitted to the workshop track of ICLR 2015.
    http://arxiv.org/abs/1412.5335


#### TODO:
- Display some basic stats on the different candidates
    - Total words
    - unique words
    - unique lemmas
    - common phrases/ngrams/noun chunks/syntactic ngrams?
    - common entities
    - pos density
    - sentence length
- Site assets
    - Bootstrap
    - Candidate pictures
    - Description of project
- Create ensemble classifier with:
    - Naive Bayes (sk-learn?)
    - doc2vec (gensim + sklearn)
- Prettify site
    - Some kind of roulette scroller while it's doing classification...?
- Show findings and results from all classifiers along with results
- Keep a list of all queries and show some of the best ones on the site
- Save/Load for ngram model
- Go through ngram issues on NLTK git

#### DONE:
- Process all text to static files
- Process user input text
- Copy new debates
- Basic Flask site with text input and on-screen string output of candidate name
