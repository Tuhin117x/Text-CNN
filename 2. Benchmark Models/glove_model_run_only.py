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
from tensorflow import keras
from sklearn.metrics import f1_score


path='E:\Storage Access\Workspaces\AMEX\Glove'
os.chdir(path)
df=pd.read_csv("train.csv")
df.drop(df.loc[df['imagename']=='image_0822'].index, inplace=True)
df.drop(df.loc[df['imagename']=='image_0867'].index, inplace=True)
df.dropna(inplace=True)
train=df.iloc[:35000,]
test=df.iloc[35000:,]


train_df=train.copy()
encoder = LabelEncoder()
train_tags=train_df['label']
encoder.fit(train_df['label'])


model_glove = keras.models.load_model("CNN_LSTM_Glove")



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
prob_predictions=model_glove.predict(test_data)
print(prob_predictions)
print(type(prob_predictions))

print("\n")
print("****************************************************************************")
print("*******   Export Model output as csv along with F1 scores           ********")
print("****************************************************************************")
print("\n")



flat_list = []
for sublist in prob_predictions:
    for item in sublist:
        flat_list.append(item)

output_df=test_df
output_df['predicted_label']=encoder.inverse_transform(output)
output_df['actual_label']=test[['label']]
#output_df['pred_prob']=prob_predictions
output_df['pred_label_untransformed']=output
f1_score_value=f1_score(output_df['actual_label'],output_df['predicted_label'],average='weighted')
output_df['f1_score']=f1_score_value
output_df['model_name']='CNN_LSTM'
print(output_df)

output_df.to_csv('predictions_2.csv',index=False)
prob_df=pd.DataFrame(prob_predictions,columns=['0','1','2','3','4','5','6','7','8','9'])
prob_df.to_csv('prediction_probabilities.csv',index=False)


