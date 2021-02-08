import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# =============================================================================
# anger_path1 = 'http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/anger-ratings-0to1.dev.gold.txt'
# anger_path2 = 'http://saifmohammad.com/WebDocs/EmoInt%20Test%20Data/anger-ratings-0to1.test.target.txt'
# fear_path = 'http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/fear-ratings-0to1.train.txt'
# joy_path = 'http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/joy-ratings-0to1.train.txt'
# sad_path1 = 'http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/sadness-ratings-0to1.train.txt'
# sad_path2 = 'http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/sadness-ratings-0to1.test.gold.txt'
# 
# 
# anger1 = pd.read_csv(anger_path1 , sep='\t', names = ['1', 'message', 'label', '4'])
# anger2 = pd.read_csv(anger_path2 , sep='\t', names = ['1', 'message', 'label', '4'])
# fear = pd.read_csv(fear_path , sep='\t', names = ['1', 'message', 'label', '4'])
# joy = pd.read_csv(joy_path , sep='\t', names = ['1', 'message', 'label', '4'])
# sad1 = pd.read_csv(sad_path1 , sep='\t', names = ['1', 'message', 'label', '4'])
# sad2 = pd.read_csv(sad_path2 , sep='\t', names = ['1', 'message', 'label', '4'])
# =============================================================================


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

out = dataset.iloc[:,:].values

np.random.shuffle(out)

#create corpus

sentences = out[:,0]

corpus = []
ps = PorterStemmer()

for i in sentences:
    temp = re.sub('[^a-zA-Z]', ' ', i)
    temp = temp.lower()
    temp = nltk.word_tokenize(temp)
    temp = [ps.stem(word)for word in temp if word not in set(stopwords.words('english'))]
    temp = ' '.join(temp)
    corpus.append(temp)

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

model = make_pipeline(CountVectorizer(max_features = 5000), MultinomialNB())

model.fit(corpus, y)

dict_emotions = {
    0: 'sadness',
    1: 'fear',
    2: 'joy',
    3: 'angry',
    }

def get_emotion(txt):
    temp = re.sub('[^a-zA-Z]', ' ', txt)
    temp = temp.lower()
    temp = nltk.word_tokenize(temp)
    temp = [ps.stem(word)for word in temp if word not in stopwords.words('english')]
    temp = ' '.join(temp)

    emotion_index = model.predict([temp])

    return dict_emotions[emotion_index[0]]

