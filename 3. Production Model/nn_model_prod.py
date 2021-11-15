import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras.utils import np_utils
from sklearn.metrics import f1_score

#--------------------------------------
# Read Datasets
#--------------------------------------

path='E:\Storage Access\Workspaces\AMEX\TextCNN'
os.chdir(path)
df=pd.read_csv("train.csv")
df.drop(df.loc[df['imagename']=='image_0822'].index, inplace=True)
df.drop(df.loc[df['imagename']=='image_0867'].index, inplace=True)

#--------------------------------------
# Data Pre-processing
#--------------------------------------


df.dropna(inplace=True)
train=df.iloc[:35000,]
test=df.iloc[35000:,]
X=train[['description']]
y=train['label']
x_test1=test[['description']]
x_test=test[['description']]
y_test=test[['label']]
y_test1=test[['label']]


train_posts = X.description.tolist()
train_tags = y
print(train_posts)
test_posts = x_test.description.tolist()
test_tags = y_test


#--------------------------------------
# Tokenizing Text Data
#--------------------------------------


max_words = 10000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
print(x_train)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
print(y_train)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
print(num_classes)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)



batch_size = 32
epochs = 50

#--------------------------------------
# Model Build
#--------------------------------------
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(x_train)          
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)



#--------------------------------------
# Model Predictions
#--------------------------------------

plt.title("Loss Curve")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('Loss_Curve.png')
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig('Accuracy_Curve.png')
plt.show();




output=model.predict_classes(x_test)
print(output)
output_df=x_test1
output_df['predicted_label']=encoder.inverse_transform(output)
output_df['actual_label']=y_test1
f1_score_value=f1_score(output_df['actual_label'],output_df['predicted_label'],average='weighted')
output_df['f1_score']=f1_score_value
output_df['model_name']='NeuralNet'
print(output_df)
output_df.to_csv('model_results.csv',index=False)
print('Test accuracy:', score[1])