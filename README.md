# Text-CNN
Recently, American Express as part of its talent search initiative organized a Data Science competition across the major campuses in India. 
The business objective behind doing that was to achieve digitization of invoicing processes such as automating expenses and for ensuring compliance and regulation.

![alt text](https://tuhin2nitdgp.files.wordpress.com/2021/11/image.png)

The objective of the competition was to extract information from structured text documents such as invoices and business documents and classify textual sections of the document into pre-defined categories such as total tax/tip/price/etc. Our Team DeltaNeutral achieved a top leaderboard position in the competition

## Model Description

For this competition, we used a deep learning model for classification by using the Keras library.For our text classification model, we plan to deploy a 7-layer deep convolutional neural network which will utilize Rectified Linear Unit activation functions within the intermediate layers while deploying a SoftMax activation function in the final layer. This was done because of our problem statement where-in we are dealing with multi-class classification instead of binary classification. We also add Drop Out layers in between the hidden layers to avoid over-fitting of our textual data.

The loss function being used will be the Categorical Cross Entropy function and we aim to achieve a low error rate by using the F1 score as a measure of accuracy. The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.

![alt text](https://tuhin2nitdgp.files.wordpress.com/2021/11/accuracy_curve.png)

We use the Loss Curves and the Accuracy curves to determine what the ideal parameters for the model are. We can clearly see that after a certain threshold the Loss starts increasing with overfitting problems becoming more severe. Hence, we make a case that having such a high epoch value is not that beneficial in terms of training our model resulting in excessive time consumption and leaving some room for more fine-tuning of our CNN.

## Model Results

For benchmarking our model, we run other standard classification models from the Sklearn library by using its pipeline functionality. We train these models first using only the text data and next by incorporating the co-ordinates of the bounding box of each textual component of the invoice. To our surprise, we find that the accuracy of these models decline on using the Layout information. In order to better incorporate structure + text information as a feature space, we also went ahead and incorporated an LSTM model with an input layer having Glove based word embeddings. You can download the glove embeddings from the Standford NLP team’s webpage, however, do note that these embedding files are very large, often times expanding up to five gigabytes of space.

![alt text](https://tuhin2nitdgp.files.wordpress.com/2021/11/modelbenchmarks.jpg)

## Accessing the Code-Base

<li>All the datasets used in this challenge are stored in this repository within the Datasets folder
<li>The python scripts used for pre-processing the JSON files into csv format is present within the Pre-Processing folder
<li>The benchmark models used for assessing relative accuracy are stored with the Benchmark Models folder
<li>The final CNN model used for predicting Round 1 and Round 2 results is stored within the Production Model folder  
