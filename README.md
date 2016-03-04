Candidate Classifier
======

This is a small personal project to showcase some of what I've learned about Natural Language processing and web-app design.




### Citations:
The debate transcriptions were taken from The American Presidency Project: `http://www.presidency.ucsb.edu`.

The NgramModel class is resurrected from the NLTK project and is mostly functional but still needs some work.


### TODO:
- Site assets
    - Bootstrap
    - Candidate pictures
    - Description of project
- Add phrases model and nltk mutli-word-expression tokenizer
- Create ensemble classifier with:
    - Naive Bayes (sk-learn?)
    - doc2vec (gensim + sklearn)
- Prettify site
    - Some kind of roulette scroller while it's doing classification...?
- Keep a list of all queries and show some of the best ones on the site
- Save/Load for ngram model
- Go through ngram issues on NLTK git

### DONE:
- Process all text to static files
- Process user input text
- Copy new debates
- Basic Flask site with text input and on-screen string output of candidate name
