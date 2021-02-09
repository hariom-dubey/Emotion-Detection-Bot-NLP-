#importing libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


#loading datasets
anger1 = pd.read_csv('anger1.csv')
anger2 = pd.read_csv('anger2.csv')
fear = pd.read_csv('fear.csv')
joy = pd.read_csv('joy.csv')
sad1 = pd.read_csv('sad1.csv')
sad2 = pd.read_csv('sad2.csv')

emotions = [anger1, anger2, fear, joy, sad1, sad2]
values = [60, 760, 820, 820, 786, 34]
dataset = pd.DataFrame()
temp1 = []
temp2 = []

for idx, emotion in enumerate(emotions):
    for i in range(values[idx]):
        temp1.append(emotion['message'][i])
        temp2.append(emotion['label'][i])

dataset['message'] = temp1
dataset['label'] = temp2

#shuffling the data
np.random.shuffle(dataset.iloc[:,:].values)

sentences = dataset.iloc[:,0]
#data preprocessing
corpus = []
ps = PorterStemmer()

for i in sentences:
    temp = re.sub('[^a-zA-Z]', ' ', i)
    temp = temp.lower()
    temp = nltk.word_tokenize(temp)
    temp = [ps.stem(word)for word in temp if word not in set(stopwords.words('english'))]
    temp = ' '.join(temp)
    corpus.append(temp)

#changing labels in the dataset to numbers
def change_label(value):
    if value == 'sadness':
        return 0
    elif value == 'fear':
        return 1
    elif value == 'joy':
        return 2
    else:
        return 3

dataset['label2'] = dataset['label'].apply(change_label)

y = dataset['label2'].values

#applying BOG
cv = CountVectorizer(max_features = 5000)
x = cv.fit_transform(corpus).toarray()

#splitting the data
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 0)

#applying naive bayes classifier
nb = MultinomialNB()

nb.fit(train_x, train_y)

y_pred = nb.predict(test_x)

mat = confusion_matrix(test_y, y_pred)
score = accuracy_score(test_y, y_pred)

print(score)

