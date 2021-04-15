#!/usr/bin/env python
# coding: utf-8

# In[1]:


#natural language toolkit
pip install nltk


# In[64]:


get_ipython().system('pip install seaborn')


# In[45]:


get_ipython().system('pip install scikit-learn')


# In[2]:


get_ipython().system('pip install gensim')


# In[ ]:





# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


# In[70]:


#Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


# In[ ]:





# In[3]:


import seaborn as sns


# In[38]:


#loading the data
stock_df=pd.read_csv('stock_sentiment.csv')


# In[39]:


stock_df


# In[40]:


stock_df.info()


# In[41]:


sns.countplot(stock_df['Sentiment'])


# In[42]:


import string
string.punctuation


# In[43]:


#Function for punctuation removal
def remove_punc(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)

    return Test_punc_removed_join


# In[44]:


stock_df['text without punctuation']=stock_df['Text'].apply(remove_punc)


# In[45]:


stock_df


# In[14]:


#REMOVING STOPWORDS


# In[46]:


nltk.download("stopwords")


# In[47]:


stop_words=stopwords.words('english')
stop_words


# In[48]:


stop_words.extend(['from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year'])


# In[49]:


stop_words


# In[50]:


# Remove stopwords and remove short words (less than 2 characters)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) >= 3 and token not in stop_words:
            result.append(token)
            
    return result


# In[51]:


stock_df['Text without punctuations and stopwords']=stock_df['text without punctuation'].apply(preprocess)


# In[52]:


stock_df


# In[53]:


# join the words into a string
stock_df['Text Without Punc & Stopwords Joined'] = stock_df['Text without punctuations and stopwords'].apply(lambda x: " ".join(x))


# In[54]:


stock_df


# In[17]:


nltk.download('punkt')


# In[55]:


# word_tokenize is used to break up a string into words
print(stock_df['Text Without Punc & Stopwords Joined'][0])
print(nltk.word_tokenize(stock_df['Text Without Punc & Stopwords Joined'][0]))


# In[56]:


tweets_length = [ len(nltk.word_tokenize(x)) for x in stock_df['Text Without Punc & Stopwords Joined'] ]
tweets_length


# In[58]:


plt.hist(tweets_length,bins=50)
plt.show()


# In[39]:


#Tokenizing and padding the data


# In[59]:


list_of_words=[]
for i in stock_df['Text without punctuations and stopwords']:
    for j in i:
        list_of_words.append(j)


# In[60]:


list_of_words


# In[61]:


# The total number of unique words in the dataset
total_words = len(list(set(list_of_words)))
total_words


# In[67]:


#Splitting into test and train data
X=stock_df['Text without punctuations and stopwords']
y=stock_df['Sentiment']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[71]:


# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(X_train)

# Training data
train_sequences = tokenizer.texts_to_sequences(X_train)

# Testing data
test_sequences = tokenizer.texts_to_sequences(X_test)


# In[72]:


train_sequences


# In[30]:


test_sequences


# In[73]:


# Add padding to training and testing
padded_train = pad_sequences(train_sequences, maxlen = 15, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 15, truncating = 'post')

for i,doc in enumerate(padded_train[:3]):
    print("The paddedencoding for ",i+1,"is",doc)


# In[75]:


# Convert the data to categorical 2D representation
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)


# In[78]:


y_train_cat.shape


# In[79]:


y_train_cat


# In[34]:


# Sequential Model
model = Sequential()

# embedding layer
model.add(Embedding(total_words, output_dim = 512))

# Bi-Directional RNN and LSTM
model.add(LSTM(256))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()


# In[37]:


# train the model
model.fit(padded_train, y_train_cat, batch_size = 32, validation_split = 0.2, epochs = 2)


# In[38]:


#Testing the model using testing dataset
# make prediction
pred = model.predict(padded_test)
pred


# In[40]:


# make prediction
prediction = []
for i in pred:
  prediction.append(np.argmax(i))

prediction


# In[41]:


# list containing original values
original = []
for i in y_test_cat:
  original.append(np.argmax(i))

original


# In[42]:


# acuracy score on text data
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(original, prediction)
accuracy


# In[43]:


# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot = True)


# In[ ]:




