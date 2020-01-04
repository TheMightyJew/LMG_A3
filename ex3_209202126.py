import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from datetime import datetime
import pytorch
import random

# tokenizer to work later
tfidfVectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
tokenizer = tfidfVectorizer.build_tokenizer()
# save some important dates in order to classify later
election_date = datetime.strptime('2016-11-16 00:00:00', '%Y-%m-%d %H:%M:%S')
iphone_date = datetime.strptime('2017-04-01 00:00:01', '%Y-%m-%d %H:%M:%S')

# A function that reads the data from the train file, extract the features and splits it to train and test sets.
def get_train_data():
    trump_tweets = []
    trump_features = []
    assistents_tweets = []
    assistents_features = []

    for line in open("trump_train.tsv"):
        splitted = line.replace('\n', '').split('\t')
        tweet, class_target, tweet_data = classify_tweet(splitted)
        if class_target == 0:
            trump_tweets.append(tweet)
            trump_features.append(tweet_data)
        else:
            assistents_tweets.append(tweet)
            assistents_features.append(tweet_data)

    trump_dict = get_dict(trump_tweets)
    assistents_dict = get_dict(assistents_tweets)
    trump_tweets_p = get_p(trump_tweets, trump_dict, assistents_dict)
    assistents__tweets_p = get_p(assistents_tweets, trump_dict, assistents_dict)

    for index in range(len(trump_features)):
        t_p, a_p = trump_tweets_p[index]
        trump_features[index].append(t_p)
        trump_features[index].append(a_p)
    for index in range(len(assistents_features)):
        t_p, a_p = assistents__tweets_p[index]
        assistents_features[index].append(t_p)
        assistents_features[index].append(a_p)

    trump_features, assistents_features = equal_class_sizes(trump_features, assistents_features)
    X_train, X_test, y_train, y_test = split_data(trump_features, assistents_features, test_size=0.2)
    return X_train, X_test, y_train, y_test

# load the pytorch network is it was the best model
def load_best_model():
    model = pytorch.Pytorch()
    model.load_state_dict(torch.load('best_model.pt'))
    return model

# trains the pytorch network is it was the best model
def train_best_model():
    X_train, X_test, y_train, y_test = get_train_data()
    torch = pytorch.Pytorch()
    torch.train(X_train, y_train)
    return torch

# reads the data from fn and extracts the features, uses the m model to predict the class. Returns an array of 0's and 1's.
def predict(m, fn):
    tweets_features = []
    tweets = []
    for line in open(fn):
        splitted = line.replace('\n', '').split('\t')
        tweet_date = datetime.strptime(splitted[2], '%Y-%m-%d %H:%M:%S')
        day = tweet_date.weekday()
        hour = tweet_date.hour
        tweet_length = len(splitted[1])
        words_num = len(splitted[1].split(" "))
        tweet_data = [day, hour, tweet_length, words_num]
        tweets_features.append(tweet_data)
        tweets.append(splitted[1])

    trump_tweets = []
    assistents_tweets = []

    for line in open("trump_train.tsv"):
        splitted = line.replace('\n', '').split('\t')
        tweet, class_target, tweet_data = classify_tweet(splitted)
        if class_target == 0:
            trump_tweets.append(tweet)
        else:
            assistents_tweets.append(tweet)

    trump_dict = get_dict(trump_tweets)
    assistents_dict = get_dict(assistents_tweets)
    tweets_p = get_p(tweets, trump_dict, assistents_dict)
    for index in range(len(tweets_features)):
        t_p, a_p = tweets_p[index]
        tweets_features[index].append(t_p)
        tweets_features[index].append(a_p)
    ans = m.predict(tweets_features)
    predictions = []
    for i in ans:
        if '0' in str(i):
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions

# classifies a tweet by set of rules, returns the tweet data, tweet classification and most of the features.
def classify_tweet(splitted):
    tweet_date = datetime.strptime(splitted[3], '%Y-%m-%d %H:%M:%S')
    day = tweet_date.weekday()
    hour = tweet_date.hour
    tweet_length = len(splitted[2])
    words_num = len(splitted[2].split(" "))
    tweet_data = [day, hour, tweet_length, words_num]
    tweet = ' '.join(tokenizer(splitted[2]))
    if splitted[1] == 'PressSec':
        class_target = 1
    elif splitted[1] == 'POTUS' and tweet_date < election_date:
        class_target = 1
    elif splitted[4] != 'android' and splitted[4] != 'iphone':
        class_target = 1
    elif splitted[4] == 'iphone' and tweet_date < iphone_date:
        class_target = 1
    else:
        class_target = 0
    return tweet, class_target, tweet_data

# function that throw away some of the data to make equal size of classification sets.
def equal_class_sizes(a, b):
    random.shuffle(a)
    random.shuffle(b)
    if len(a) > len(b):
        a = a[:len(b)]
    elif len(b) > len(a):
        b = b[:len(a)]
    return a, b

# function that split the data and shuffle it to train and test sets.
def split_data(trump_tweets, assistents_tweets, test_size):
    zeroes_list = [0] * len(trump_tweets)
    oness_list = [1] * len(assistents_tweets)
    trump_X_train, trump_X_test, trump_y_train, trump_y_test = train_test_split( trump_tweets, zeroes_list, test_size=test_size, random_state=0)
    assistents_X_train, assistents_X_test, assistents_y_train, assistents_y_test = train_test_split( assistents_tweets, oness_list, test_size=test_size, random_state=0)
    train_tmp = list(zip(trump_X_train + assistents_X_train, trump_y_train + assistents_y_train))
    test_tmp = list(zip(trump_X_test + assistents_X_test, trump_y_test + assistents_y_test))
    random.shuffle(train_tmp)
    random.shuffle(test_tmp)
    X_train, y_train = zip(*train_tmp)
    X_test, y_test = zip(*test_tmp)
    return X_train, X_test, y_train, y_test

# return a dictionary of idf and words for a given text
def get_dict(tweets):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer.fit_transform(tweets)
    return tfidf_vectorizer.vocabulary_

# returns an array of average idf for each tweet is given both for trump tweets and assistannts tweets
def get_p(tweets, trump_dict, assistents_dict):
    p = []
    for tweet in tweets:
        score_trump = 0
        score_assistents = 0
        counter = 0
        for token in tweet.split(' '):
            if token in trump_dict.keys() and trump_dict[token] != 0:
                score_trump += trump_dict[token]
            else:
                score_trump += 1 / len(trump_dict.keys())
            if token in assistents_dict.keys() and assistents_dict[token] != 0:
                score_assistents += assistents_dict[token]
            else:
                score_assistents += 1 / len(assistents_dict.keys())
            counter += 1
        p.append((score_trump / counter,score_assistents / counter) )
    return p

# function to test all algorithms
def test_all_algo():
    X_train, X_test, y_train, y_test = get_train_data()

    logistic_regression = LogisticRegression(random_state=0).fit(X_train, y_train)

    svc = SVC(gamma = 'auto').fit(X_train, y_train)
    linear_svc = LinearSVC().fit(X_train, y_train)

    torch_model = pytorch.Pytorch()
    torch_model.train(X_train, y_train)

    KNN_model = KNeighborsClassifier(n_neighbors=5)
    KNN_model.fit(X_train, y_train)

    lr_score = round(logistic_regression.score(X_test, y_test)*100,2)
    svc_score = round(svc.score(X_test, y_test)*100,2)
    l_svc_score = round(linear_svc.score(X_test, y_test)*100,2)
    pytorch_score = round(torch_model.score(X_test, y_test)*100,2)
    KNN_score = round(KNN_model.score(X_test, y_test)*100, 2)
    return lr_score, svc_score, l_svc_score, pytorch_score, KNN_score

# function that runs all algorithms few time and print averages accuracy.
def test_avareges(times):
    lr_sum = 0
    svc_sum = 0
    l_svc_sum = 0
    pytorch_sum = 0
    KNN_sum = 0
    for i in range(times):
        lr_score, svc_score, l_svc_score, pytorch_score, KNN_score = test_all_algo()
        lr_sum += lr_score
        svc_sum += svc_score
        l_svc_sum += l_svc_score
        pytorch_sum += pytorch_score
        KNN_sum += KNN_score
    print('Averages after ' + str(times) + ' iterations: ')
    print('Logistic_Regression average: ' + str(round(lr_sum/times, 2)) + ' %')
    print('SVC                 average: ' + str(round(svc_sum/times, 2)) + ' %')
    print('Linear_SVC          average: ' + str(round(l_svc_sum/times, 2)) + ' %')
    print('Pytorch             average: ' + str(round(pytorch_sum/times, 2)) + ' %')
    print('KNN                 average: ' + str(round(KNN_sum/times, 2)) + ' %')

# function that write predictions to the test file.
def write_sol():
    predictions = predict(load_best_model(),'trump_test.tsv')
    f = open("209202126.txt", "a")
    for prediction in predictions:
        f.write(str(prediction) + ' ')
    f.close()

def main():
    test_avareges(10)

if __name__ == '__main__':
    main()
