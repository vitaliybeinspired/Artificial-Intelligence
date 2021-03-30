# Artificial Intelligence



![](https://github.com/vitaliybeinspired/Artificial-Intelligence/blob/main/images/surestart.png)


## Responses

# DAY 1 (2/8)
- I'm curious to learn all about Artificial Intelligence such as machine learning, computer vision, image processing, natural language processing, in the fields like finance and more. I would like to learn what else is out there and how to use it. And how to connect with professionals and start an internship or full-time.

# Day 2 (2/9)

1.What is the difference between supervised and unsupervised learning? 

Supervised learning is having the training data and knowing the outcome labels of those like cat or dog. Uses decision tree algorithm.
Unsupervised learning is having the training data then trying to group them into clusters to find similarity. Uses K nearest neighbor algorithm. 

2.Describe why the following statement is FALSE: Scikit-Learn has the power to visualize data without a Graphviz, Pandas, or other data analysis libraries.

Scikit Learn is focused on Machine Learning, e.g data modelling. It is not concerned with the loading, handling, manipulating, and visualising of data.

# Day 3 (2/10)

1. What are “Tensors” and what are they used for in Machine Learning? 

They are vectors with basis (1 unit of anything I’m using) and have 3 subcomponents like vector A has xyz sub components. 
They are the same from any observer and reference frame. Can have multiple directions.


2.What did you notice about the computations that you ran in the TensorFlow 
programs (i.e. interactive models) in the tutorial? 

They are all arrays. 

# Day 4 (2/11)

Problem: Predicting stock market 

Data: Huge Stock Market Dataset https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs 

Algorithm: Convolutional neural networks. 

I learned about batches would take a section of training data. The whole data is one epoch.

# Day 5 (2/12)


Sarcasim notebook that removes common words and converts them to vectors then tests which headlines are sarcastic or not.
I learned that I need to use a sequenal model and to visual data at every step to make sure that samples are balanced for example.

# Day 8 (2/15)

In the convolutional neural network, I learned that the CNN has an optional Pooling layer which increases the accuracy. The pooling layer takes a section and calculates the max and average of all the pixels in that section.

# Day 9 (2/16)

Bias in the vision and languge of AI was that it labels yellow banana instead of banana like people would because people don't say yellow because it's common. I also remember reading about the AI creating a model on past successful resumes which was biased and applying it to current people.

# Day 10 (2/17)

Convolutional Neural Network breaks up a image into smaller parts and then analyzes each feature in isolation which then make a decision about the image as a whole.

Convolutional layer filters an image, scanning a few pixels at a time and creating a feature map that predicts the class to which each feature belongs to.

Pooling layer (downsampling) keeps the most important parts of the convolutional layer. Several rounds of convolutional and pooling.

Fully Connected Neural Network is taking the results from the convolution/pooling process and use them to classify the image into a label.

Fully connected input layer takes the output of the previous layers and flattends them into a single vector.

First fully connected layer takes the inputs from the feature analysis and applies weights to predict the correct label.

Fully connected output layer gives the final probabilities for each label.
 
Practical applications include predicting labels of images.

# Day 11 (2/18)

Notebook ExploringMNIST.ipynb for classifying handwritten digits.

# Day 12 (2/19)

I learned about data augmentation and the layers in the CNN. I uploaded an image classifer for cats and dogs and changed images amounts and epochs. I learned to prefetch images to avoid I/O because loading from disk is slower than loading from memory.

# Day 15 (2/22)

Today I learned about k-folds and how kaggle vs colab notebooks

# Day 16 (2/23)

The advantages of the Rectified Linear (ReLu) activation function is that it overcomes the vanishing gradient problem and allows models to learn faster and perform better. It’s good for multiplayer perceptrons and CNN’s. It ranges from 0 to 1 output. Inputs need to be normalized and standardized. I would use the ReLu for images.

# Day 17 (2/24)

Uploaded house price prediction notebook about loss functions.

# Day 18 (2/25)

I learned underfitting is a model that doesn’t generalize well and can be fixed by increasing capacity like adding more training examples, nodes and layers. Overfitting is a model that does too well on the training data but not on test data. This can be fixed with dropout and stopping when validation loss starts increasing. Weights need to be small because large weighs cause sharp transitions in the activation functions and result in large changes in output for small changes in input.

I changed the house_price_prediction.ipynb to test changing loss from binary_crossentropy to mean_squared_error. The result was that binary_crossentropy accuracy was 87% for 10 epochs and mean_squared_error was 51% accuracy for 10 epochs

# Day 19 (2/26)

I learned about Upsampling and tranposing 2dconv and autoencoders. 
I learned about GAN networks how they create real and fake data and try to disciminate which one it is.
Uploaded two notebooks.

# DAY 22 (3/1) - NATURAL LANGUAGE PROCESSING (NLP)

I used flair for NLP to detect Named Entity Recognition. 
The GPT-2 model from openAI created a paragraph story out of a sentence. Writers may use this to generate blogs or even write books, completely replacing authors.

# Day 23
Add emotion detection model that uses OpenCV (Computer Vision)

# Day 24
Adding speech emotion analyzer folder from Mitesh Puthran. I learned about using pyaudio to record self in python.

# Makeathon
2nd place winner! Team Aither. 
