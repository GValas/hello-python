#  https://hackernoon.com/how-to-build-a-simple-spam-detecting-machine-learning-classifier-4471fe6b816e

import os
from enum import Enum
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split


class MessageType(Enum):
    HAM = 'ham'
    SPAM = 'spam'


class DataLoader:
    @staticmethod
    def load():
        cwd = os.getcwd()
        csv = os.path.join(cwd, "spam.csv")
        df = pd.read_csv(csv, usecols=['v1', 'v2'])
        df.columns = ['spam', 'sms']
        return train_test_split(df['sms'], df['spam'], test_size=0.33)


#  events definition : A = spam, B = context (bag of words)
class SpamClassifier:
    def __init__(self):
        self.numWords = 0
        self.words_counts = {MessageType.SPAM: {}, MessageType.HAM: {}}
        self.types_counts = {MessageType.SPAM: 0, MessageType.HAM: 0}
        self.negativeTotal = 0
        self.positiveTotal = 0
        self.alpha = 1  # laplace smoothing
        self.p_spam = 0
        self.nlp = spacy.load('en')

    # creating a spam model
    # runs once on a training data
    def train(self, messages: list, labels: list):

        # p(spam)
        self.p_spam = labels.count(MessageType.SPAM.value) / float(len(messages))

        # p(words/spam) = p(w1,...,w2/spam) = p(w1/spam)...p(wn/spam)
        for msg, lbl in zip(messages, labels):
            words = [str(word) for word in self.nlp(msg)]
            for word in words:
                self.words_counts[MessageType(lbl)][word] = self.words_counts[MessageType(lbl)].get(word, 0) + 1
                self.types_counts[MessageType(lbl)] += 1

    def validate(self, messages: list, labels: list):
        i = 0
        for msg, lbl in zip(messages, labels):
            if self.classify(msg) is MessageType(lbl):
                i = i + 1
        a = i / float(len(messages))
        print("accuracy={}".format(a))

        # p(spam/words) > p(ham/words) => p(words/spam) * p(spam) > p(words/ham) * p(ham)

    def classify(self, message):
        isSpam = self.p_spam * self.__conditionalEmail(message, MessageType.SPAM)  # P (A | B)
        notSpam = (1 - self.p_spam) * self.__conditionalEmail(message, MessageType.HAM)  # P(¬A | B)
        return MessageType.SPAM if isSpam > notSpam else MessageType.HAM

        # p(msg/spam) = p(w1/spam)...p(wn/spam)

    # p(words/spam) = p(w1,...,w2/spam) = p(w1/spam)...p(wn/spam)
    def __conditionalEmail(self, message, label):
        p = 1.0
        words = [str(word) for word in self.nlp(message)]
        for word in words:
            p *= (self.words_counts[label].get(word, 0) + self.alpha) / float(
                self.types_counts[label] + self.alpha * self.numWords)
        return p


X_train, X_test, y_train, y_test = DataLoader().load()
clf = SpamClassifier()
clf.train(X_train.tolist(), y_train.tolist())
clf.validate(X_test.tolist(), y_test.tolist())
