import random

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from datetime import datetime
from sklearn.model_selection import train_test_split
import pytorch
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
import nltk
import numpy as np

data = []
target = []
election_date = datetime.strptime('2016-11-16 00:00:00','%Y-%m-%d %H:%M:%S')
iphone_date = datetime.strptime('2017-04-01 00:00:01','%Y-%m-%d %H:%M:%S')
trump_tweets = []
assistents_tweets = []
for line in open("trump_train.tsv"):
    splitted = line.replace('\n','').split('\t')
    tweet_date = datetime.strptime(splitted[3],'%Y-%m-%d %H:%M:%S')
    # data
    day = tweet_date.weekday()
    hour = tweet_date.hour
    tweet_length = len(splitted[2])
    words_num = len(splitted[2].split(" "))
    tweet_data = [day, hour, tweet_length, words_num]
    data.append(tweet_data)
    if splitted[1] == 'PressSec':
        class_target = 1
    elif splitted[1] == 'POTUS' and tweet_date < election_date:
        class_target = 1
    elif splitted[4] != 'iphone':
        class_target = 0
    elif splitted[4] == 'iphone' and tweet_date < iphone_date:
        class_target = 1
    else:
        class_target = 0
    target.append(class_target)
    """
    tweet = nltk.word_tokenize(splitted[2])
    if target == 0:
        trump_tweets.append(tweet)
    else:
        assistents_tweets.append(tweet)
    """

"""
train, vocab = padded_everygram_pipeline(2, trump_tweets)
lm = nltk.lm.MLE(4)
lm.fit(train, vocab)
for line in open("trump_train.tsv"):
    splitted = line.replace('\n','').split('\t')
    t = lm.entropy(splitted[2])
"""

def fix(X_train, y_train):
    count0 = 0
    count1 = 0
    for target in y_train:
        if target == 0:
            count0 += 1
        else:
            count1 += 1
    if count0 > count1:
        for i in range(count0 - count1):
            for index in range(len(X_train)):
                if y_train[index] == 0:
                    X_train = X_train[:index] + X_train[index + 1:]
                    y_train = y_train[:index] + y_train[index + 1:]
                    break
    else:
        for i in range(count1 - count0):
            for index in range(len(X_train)):
                if y_train[index] == 1:
                    X_train = X_train[:index] + X_train[index + 1:]
                    y_train = y_train[:index] + y_train[index + 1:]
                    break
    tmp = list(zip(X_train, y_train))
    random.shuffle(tmp)
    X_train, y_train = zip(*tmp)
    return X_train, y_train

X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.2, random_state=0)
X_train, y_train = fix(X_train, y_train)



logistic_regression = LogisticRegression(random_state=0).fit(X_train, y_train)
print('logistic_regression accuracy: ' + str(round(logistic_regression.score(X_test, y_test)*100,2)) + ' %')

svc = SVC(gamma = 'auto').fit(X_train, y_train)
print('svc                 accuracy: ' + str(round(svc.score(X_test, y_test)*100,2)) + ' %')

linear_svc = LinearSVC().fit(X_train, y_train)
print('linear_svc          accuracy: ' + str(round(linear_svc.score(X_test, y_test)*100,2)) + ' %')

torch = pytorch.Pytorch()
torch.train(X_train, y_train)
print('pytorch             accuracy: ' + str(round(torch.score(X_test, y_test)*100,2)) + ' %')






