import gensim
# loading the downloaded model
model = gensim.models.KeyedVectors.load_word2vec_format('/home/gege/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

# the model is loaded. It can be used to perform all of the tasks mentioned above.

# getting word vectors of a word
dog = model['dog']

# performing king queen magic
print(model.most_similar(positive=['woman', 'king'], negative=['man']))

# picking odd one out
print(model.doesnt_match("breakfast cereal dinner lunch".split()))

# printing similarity index
print(model.similarity('woman', 'man'))
