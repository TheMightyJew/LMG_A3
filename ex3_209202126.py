import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from datetime import datetime
from sklearn.model_selection import train_test_split
import pytorch

def classify_tweet(splitted):
    tweet_date = datetime.strptime(splitted[3], '%Y-%m-%d %H:%M:%S')
    # data
    day = tweet_date.weekday()
    hour = tweet_date.hour
    tweet_length = len(splitted[2])
    words_num = len(splitted[2].split(" "))
    tweet_data = [day, hour, tweet_length, words_num]
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
    return tweet_data, class_target

def fix(a, b):
    random.shuffle(a)
    random.shuffle(b)
    if len(a) > len(b):
        a = a[:len(b)]
    elif len(b) > len(a):
        b = b[:len(a)]
    return a, b

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


election_date = datetime.strptime('2016-11-16 00:00:00','%Y-%m-%d %H:%M:%S')
iphone_date = datetime.strptime('2017-04-01 00:00:01','%Y-%m-%d %H:%M:%S')
trump_tweets = []
assistents_tweets = []
for line in open("trump_train.tsv"):
    splitted = line.replace('\n', '').split('\t')
    tweet_data, class_target = classify_tweet(splitted)
    if class_target == 0:
        trump_tweets.append(tweet_data)
    else:
        assistents_tweets.append(tweet_data)
trump_tweets, assistents_tweets = fix(trump_tweets, assistents_tweets)
X_train, X_test, y_train, y_test = split_data( trump_tweets, assistents_tweets, test_size=0.2)

logistic_regression = LogisticRegression(random_state=0).fit(X_train, y_train)

svc = SVC(gamma = 'auto').fit(X_train, y_train)

linear_svc = LinearSVC().fit(X_train, y_train)

torch = pytorch.Pytorch()
torch.train(X_train, y_train)

print('logistic_regression accuracy: ' + str(round(logistic_regression.score(X_test, y_test)*100,2)) + ' %')
print('svc                 accuracy: ' + str(round(svc.score(X_test, y_test)*100,2)) + ' %')
print('linear_svc          accuracy: ' + str(round(linear_svc.score(X_test, y_test)*100,2)) + ' %')
print('pytorch             accuracy: ' + str(round(torch.score(X_test, y_test)*100,2)) + ' %')






