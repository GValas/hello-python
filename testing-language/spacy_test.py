import spacy

nlp = spacy.load('en')
doc1 = nlp(u"this's spacy tokenize test")
print(doc1)
for token in doc1:
    print(token)