#!/usr/bin/env python
# coding: utf-8

# ## MODEL TRAINING PART

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import pyaudio
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix


def feature_selection():    

    data = pd.read_csv("dataset_path")
    data=data.clip()
    X = data.iloc[:,1:87]  #independent columns
    y = data.iloc[:,88]    #target column i.e price range
    #using SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=7)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #using two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    return featureScores.nlargest(30,'Score')

feature_selection()

df=pd.read_csv("csv_path_dataset_1")

def train_based_url():
    feature_selection()
    training_set, test_set = train_test_split(df, test_size = 0.2, random_state = 1)
    #X_train = training_set.iloc[:,0:2].values // 83,82,56,81,21 #56,81,21
    X_train = training_set.iloc[:,[83,82,56,81,70,46,69,44,74,67,49,1,85,50,9,39,42,7,20]].values
    Y_train = training_set.iloc[:,88].values
    X_test = test_set.iloc[:,[83,82,56,81,70,46,69,44,74,67,49,1,85,50,9,39,42,7,20]].values
    Y_test = test_set.iloc[:,88].values

    clf = RandomForestClassifier(max_depth=60, random_state=0)
    clf.fit(X_train,Y_train)

    y_pred=clf.predict(X_test)
    
    cm = confusion_matrix(Y_test,y_pred)
    accuracy = float(cm.diagonal().sum())/len(Y_test)
    #pint("\nAccuracy Of Random Forest For The Given Dataset : ", accuracy)
    return "Model Accuracy is : ", accuracy


train_based_url()

# ## RECCURRENT NEURAL NETWORK

url="spam_csv_path"
data = pd.read_csv(url)

texts = []
labels = []
for i, label in enumerate(data['Category']):
    texts.append(data['Message'][i])
    if label == 'ham':
        labels.append(0)
    else:
        labels.append(1)

texts = np.asarray(texts)
labels = np.asarray(labels)


print("number of texts :" , len(texts))
print("number of labels: ", len(labels))

# features
max_features = 10000
# cut off the words after seeing 500 words in each word
maxlen = 500


# 80% of data for training and 20% for validation
training_samples = int(5572 * .8)
validation_samples = int(5572 - training_samples)
# sanity check
print(len(texts) == (training_samples + validation_samples))
print("The number of training {0}, validation {1} ".format(training_samples, validation_samples))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)

print("data shape: ", data.shape)

np.random.seed(42)
# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


texts_train = data[:training_samples]
y_train = labels[:training_samples]
texts_test = data[training_samples:]
y_test = labels[training_samples:]

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)

acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training acc')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()


pred = model.predict(texts_test)
acc = model.evaluate(texts_test, y_test)
proba_rnn = model.predict(texts_test)

print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, y_test))

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history_ltsm = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)

acc = history_ltsm.history['acc']
val_acc = history_ltsm.history['val_acc']
loss = history_ltsm.history['loss']
val_loss = history_ltsm.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='purple', label='training accuracy')
plt.plot(epochs, val_acc, '-', color='green', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='purple', label='training accuracy')
plt.plot(epochs, val_loss,  '-', color='green', label='validation accuracy')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[19]:


pred = model.predict(texts_test)
acc = model.evaluate(texts_test, y_test)
proba_ltsm = model.predict(texts_test)

print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, y_test))

# ## WEBSCRAP PART

'''
print(html)
    title = re.sub("<.*?>", "", html) # Remove HTML tags
    return title
'''

def webscrap(url):
    from urllib.request import urlopen
    import re    
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    return html

htmlcontent=webscrap(WebUrl)
print(htmlcontent)

def getVideoUrl(htmlcontent):    
    url_videos=[]
    words=htmlcontent.split(" ")
    for i in words:
        if(".mp4" in i) or (".mp4" in i):
            print(i)
            url_videos=url_videos
            return url_videos

def getImageUrl(htmlcontent):
    url_images=[]
    words=htmlcontent.split(" ")
    for i in words:
        if(".jpg" in i) or (".png" in i):
            url_images=url_images+","+i
            return url_images

def getFilteredContent(htmlcontent):
    filtered_content = re.sub("<.*?>", "", htmlcontent) # Remove HTML tags
    return filtered_content

yt=getVideoUrl(htmlcontent)
print(yt)

# ## EXTRACT SPEECHTEXT FROM VIDEO

file_path=r"deep_video_path"
video=mp.VideoFileClip(file_path)

audio_file = video.audio 
audio_file.write_audiofile("audio_path")
# Initialize recognizer 
r = sr.Recognizer() 
  
# Load the audio file 
with sr.AudioFile("audio_path") as source: 
    data = r.record(source)


cont=webscrap(WebUrl)

word=cont
wd=word.split(" ")
for i in wd:
    if(".jpg" in i) or (".png" in i):
        print(i)
        


