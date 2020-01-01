# https://github.com/cytora/pycon-nlp-in-10-lines/blob/master/00_spacy_intro.ipynb

# Import spacy and English models
import spacy

nlp = spacy.load('en')


# Process sentences 'Hello, world. Natural Language Processing in 10 lines of code.' using spaCy
doc = nlp(u'Hello, world. Natural Language Processing in 10 lines of code.')


# Get first token of the processed document
token = doc[0]
print(token)

# Print sentences (one sentence per line)
for sent in doc.sents:
    print(sent)



# For each token, print corresponding part of speech tag
for token in doc:
    print('{} - {}'.format(token, token.pos_))


