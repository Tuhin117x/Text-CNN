import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import nltk
import string
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import f1_score

print("\n")
print("****************************************************************************")
print("*****Welcome to the solution script for the AMEX data science challenge*****")
print("****************************************************************************")
print("\n")


path='E:\Storage Access\Workspaces\AMEX\Glove'
os.chdir(path)
df=pd.read_csv("train.csv")
df.drop(df.loc[df['imagename']=='image_0822'].index, inplace=True)
df.drop(df.loc[df['imagename']=='image_0867'].index, inplace=True)
df.dropna(inplace=True)
train=df.iloc[:35000,]
test=df.iloc[35000:,]


train_df=train.copy()
print("The shape of training dataset is:"+str(train_df.shape))






tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['description'])
train_sequences = tokenizer.texts_to_sequences(train_df['description'])
train_data = pad_sequences(train_sequences, maxlen=300)


print("\n")
print("****************************************************************************")
print("*******Reading the Glove Embedding File to create an Embedding Index********")
print("****************************************************************************")
print("\n")

#EMBEDDING_FILE = 'glove.840B.300d.txt'
EMBEDDING_FILE = 'glove.6B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,errors="ignore"))


embedding_matrix = np.zeros((50000, 300))
for word, index in tokenizer.word_index.items():
    if index > 50000 - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


print(embedding_matrix)
print("\n")
print("****************************************************************************")
print("*******               Embedding Matrix and Indices Created          ********")
print("****************************************************************************")
print("\n")



encoder = LabelEncoder()
train_tags=train_df['label']
encoder.fit(train_df['label'])
y_train = encoder.transform(train_tags)
num_classes = np.max(y_train) + 1
print("The number of classes to be predicted is: "+str(num_classes))
y_train = np_utils.to_categorical(y_train, num_classes)


model_glove = Sequential()
model_glove.add(Embedding(50000, 300, input_length=300, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(300))
#model_glove.add(Dense(1, activation='sigmoid'))
model_glove.add(Dense(num_classes, activation='softmax'))
#model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n")
print("****************************************************************************")
print("*******   Train Classification Model (CNN wrapped by LSTM)          ********")
print("****************************************************************************")
print("\n")




model_glove.fit(train_data, y_train, validation_split=0.2, epochs = 2)
model_glove.save("CNN_LSTM_Glove")



print("\n")
print("****************************************************************************")
print("*******   Make predictions using the CNN+LSTM Model                 ********")
print("****************************************************************************")
print("\n")



test_df=test.copy()
print("The shape of test dataset is:"+str(test_df.shape))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(test_df['description'])
test_sequences = tokenizer.texts_to_sequences(test_df['description'])
test_data = pad_sequences(test_sequences, maxlen=300)

predictions=model_glove.predict_classes(test_data)
output=predictions


print("\n")
print("****************************************************************************")
print("*******   Export Model output as csv along with F1 scores           ********")
print("****************************************************************************")
print("\n")



output_df=test_df
output_df['predicted_label']=encoder.inverse_transform(output)
output_df['actual_label']=test[['label']]
f1_score_value=f1_score(output_df['actual_label'],output_df['predicted_label'],average='weighted')
output_df['f1_score']=f1_score_value
output_df['model_name']='CNN_LSTM'
print(output_df)
output_df.to_csv('predictions_1.csv',index=False)




#output_df=pd.DataFrame({"predicted_label":test_df["imagename"].values})