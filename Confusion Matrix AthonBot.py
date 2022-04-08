#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import random
import string
from sklearn.model_selection import KFold


# In[40]:


df = pd.read_csv('dataset/tweetlabels1000_labeled.xlsx - Sheet2.csv')
print(df.shape)
df


# In[41]:


# drop rows with label = campaign, own tweet, & incomplete
df = df.loc[~df['Label'].isin(['campaign','own tweet','incomplete'])].copy()
df


# In[42]:


# Label Encoding & drop columns
df['label'] = df['Label'].map({'indirect complaint': 0, 'remark': 1, 'negative remark': 2, 'direct compliment' : 3 , 'direct complaint' : 4, 'none' : 5, 'inquiry' : 6})
df.drop(['No', 'Label', 'Username'], axis=1, inplace=True)
df


# In[43]:


df.groupby( by='label').count()


# In[44]:


# to lowercase
df['tweet_lower'] = df['Tweet'].str.lower()
df


# In[45]:


def text_cleansing(text):
    
       
    # remove non ASCII (emoticon, chinese word, etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    
    # remove digits (using regex) -> subtitute
    text = re.sub('\d+', '', text)
    
    # remove punctuation, reference: https://stackoverflow.com/a/34294398
    #text = text.translate(str.maketrans('', '', string.punctuation))
    
    # remove punctuation, except mention tag @
    text = re.sub(r'[^\w\s@]', '', text)
    
    # remove whitespace in the beginning and end of sentence
    text = text.strip()
    
    # remove extra whitespace in the middle of sentence (using regex)
    text = re.sub('\s+', ' ', text)
    
    # remove url in tweet (using regex)
    text = re.sub(r"\bhttp\w+", "", text)
    
    return text


# In[46]:


# text_cleansing()
df['tweet_clean'] = df['tweet_lower'].apply(lambda x: text_cleansing(x))
df


# In[47]:


# normalization indonesian words
normalized_word = pd.read_csv('dataset/new_kamusalay.csv', header=None)
data_dict = dict(zip(normalized_word[0], normalized_word[1]))
len(data_dict)


# In[48]:


def normalize_text(text):
    return ' '.join(data_dict.get(word, word) for word in text.split())


# In[49]:


df['tweet_normalized'] = df['tweet_clean'].apply(lambda x: normalize_text(x))
df


# In[50]:


# stemming 'kata berimbuhan'
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# In[51]:


df['tweet_stemmed'] = df['tweet_normalized'].apply(lambda x: stemmer.stem(x))
df


# In[52]:


# Define a function to compute the max length of sequence
def max_length(sequences):
    '''
    input:
        sequences: a 2D list of integer sequences
    output:
        max_length: the max length of the sequences
    '''
    max_length = 0
    for i, seq in enumerate(sequences):
        length = len(seq)
        if max_length < length:
            max_length = length
    return max_length


# In[53]:


from tensorflow.keras import regularizers
from tensorflow.keras.constraints import MaxNorm

def define_model(input_dim = None, output_dim=300, max_length = None ):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=input_dim, 
                                  mask_zero= True,
                                  output_dim=output_dim, 
                                  input_length=max_length, 
                                  input_shape=(max_length, )),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=False)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=7, activation='sigmoid')
    ])
    
    model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# In[54]:


class myCallback(tf.keras.callbacks.Callback):
    # Overide the method on_epoch_end() for our benefit
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.93):
            print("\nReached 93% accuracy so cancelling training!")
            self.model.stop_training=True


callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, 
                                             patience=10, verbose=2, 
                                             mode='auto', restore_best_weights=True)


# In[55]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

# Parameter Initialization
trunc_type='post'
padding_type='post'
oov_tok = "<UNK>"

columns = ['acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9', 'acc10', 'AVG']
record = pd.DataFrame(columns = columns)

# prepare cross validation with 10 splits and shuffle = True
kfold = KFold(n_splits=10, shuffle=False, random_state=None)

# Separate the tweet and label
tweets, labels = list(df.tweet_stemmed), list(df.label)

exp=0

# kfold.split() will return set indices for each split
acc_list = []
for train, test in kfold.split(tweets):
    
    exp+=1
    print('Training {}: '.format(exp))
    
    train_x, test_x = [], []
    train_y, test_y = [], []

    for i in train:
        train_x.append(tweets[i])
        train_y.append(labels[i])

    for i in test:
        test_x.append(tweets[i])
        test_y.append(labels[i])

    # Turn the labels into a numpy array
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    
    # one-hot
    onehot_train_y = to_categorical(train_y)
    onehot_test_y = to_categorical(test_y)

    # encode data using Cleaning and Tokenization
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(train_x)

    # Turn the text into sequence
    training_sequences = tokenizer.texts_to_sequences(train_x)
    test_sequences = tokenizer.texts_to_sequences(test_x)

    max_len = max_length(training_sequences)

    # Pad the sequence to have the same size
    Xtrain = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
    Xtest = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    word_index = tokenizer.word_index
    vocab_size = len(word_index)+1

    # Define the input shape
    model = define_model(input_dim=vocab_size, max_length=max_len)

    # Train the model
    model.fit(Xtrain, onehot_train_y, batch_size=32, epochs=30, verbose=1, 
              callbacks=[callbacks], validation_data=(Xtest, onehot_test_y))

    # evaluate the model
    loss, acc = model.evaluate(Xtest, onehot_test_y, verbose=0)
    print('Test Accuracy: {}'.format(acc*100))

    acc_list.append(acc*100)

mean_acc = np.array(acc_list).mean()
entries = acc_list + [mean_acc]

temp = pd.DataFrame([entries], columns=columns)
record = record.append(temp, ignore_index=True)
print()
print(record)
print()


# In[56]:


record


# In[57]:


# define the preprocessing pipeline
def text_preprocessing(text):
    # case folding
    text = text.lower()
    
    # text cleansing
    text = text_cleansing(text)
    
    # text normalization
    text = normalize_text(text)
    
    # stemming
    text = stemmer.stem(text)
    
    return text


# In[58]:


# predicting a text
from keras.utils.np_utils import to_categorical

def text_predict(text):
    
    # preprocess
    text_cleaned = text_preprocessing(text)
    text_sequence = tokenizer.texts_to_sequences([text_cleaned])

    max_len= max_length(training_sequences)

    seq_padded = pad_sequences(text_sequence, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    # prediction
    pred_proba = model.predict(seq_padded)

    result = np.argmax(pred_proba)

    return result


# In[59]:


# prediction test
text_predict('Indihome sampah')


# In[60]:


# predicting data set

df['tweet_predict'] = df['tweet_stemmed'].apply(lambda x: text_predict(x))
df


# In[62]:


# confusion matrix  
# source: https://datatofish.com/confusion-matrix-python/

confusion_matrix = pd.crosstab(df['label'], df['tweet_predict'], rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
plt.show()

# labels:
        #0: indirect complaint
        #1: remark
        #2: negative remark
        #3: direct compliment
        #4: direct complaint
        #5: none
        #6: inquiry

