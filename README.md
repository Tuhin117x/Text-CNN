# Text-CNN
Recently, American Express as part of its talent search initiative organized a Data Science competition across the major campuses in India. 
The business objective behind doing that was to achieve digitization of invoicing processes such as automating expenses and for ensuring compliance and regulation.

![alt text](https://tuhin2nitdgp.files.wordpress.com/2021/11/image.png)

The objective of the competition was to extract information from structured text documents such as invoices and business documents and classify textual sections of the document into pre-defined categories such as total tax/tip/price/etc. Our Team DeltaNeutral achieved a top leaderboard position in the competition

For this competition, we used a deep learning model for classification by using the Keras library.For our text classification model, we plan to deploy a 7-layer deep convolutional neural network which will utilize Rectified Linear Unit activation functions within the intermediate layers while deploying a SoftMax activation function in the final layer. This was done because of our problem statement where-in we are dealing with multi-class classification instead of binary classification. We also add Drop Out layers in between the hidden layers to avoid over-fitting of our textual data.

