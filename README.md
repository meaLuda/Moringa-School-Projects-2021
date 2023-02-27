<h1 style="text-color:green">Moringa Projects 2021</h2>


## Proj_1: FakeNewsDetector Using NLP [Tensorflow, XGBoost]


```
Project entailed creating a fake news detector using NLP. The project was done in two parts. The first part involved creating a fake news detector using a Tensorflow NeuralNetwork Classifier.

This however was not successful as the accuracy was very low.

The second part involved creating a fake news detector using a XGboost classifier. The proved a better accuracy of 0.60, which is still not good enough, albeit better than the first part.

No futher work was done on this project. or shall be done it was purely for learning purposes.
```

---

## Proj_2: Image Classification [Pytorch, CNN]


```
This project entailed learning how to manipulate images and create a classifier using CNN.

The first part I worked on visualizing image data, reducing its dimensionality and clustering it.
The DataSets included celebrities  download from the Internet from the early 2000s.  

The tasks included:
1.  Write your own version of KNN (k=1) where you use the SSD (sum of squared differences) to compute similarity

2.  Verify that your KNN has a similar accuracy as sklearn’s version

3.  Standardize your data (zero mean, divide by standard deviation)

4.  Reduces the data to 100D using PCA

5.  Compute the KNN again where K=1 with the 100D data.  Report the accuracy

6.  Compute the KNN again where K=1 with the 100D Whitened data.  Report the accuracy

7.  Reduces the data to 2D using PCA

8.  Graphs the data for visualization

```

### check Notebook for more details
[Notebook](02_ImagePredictions_with_NeuralNetworks\Image_Dimensionality_Reduction_&_Clustering.ipynb)

```
In the second part I created a CNN classifier using Pytorch. 

The goal was The goal of this project is to train a regression Neural Network to predict the value 
of a pixel I(x, y) given its coordinates (x,y). We will use the square loss functions on
the training examples L(y, f(x)) = (y - f(x))2

a) Train a NN with one hidden layer containing 128 neurons, followed by ReLU.
Train the NN for 300 epochs using the square loss 

    (1). Use the SGD optimizer
        with minibatch size 64, and an appropriate learning rate (e.g. 0.003). 
        Reduce the learning rate to half every 100 epochs. Show a plot of the loss function vs epoch
        number. Display the image reconstructed from the trained NN fy (4,7),7 €
        {1,...,84},5 € {1,...,128}. (2 points)

b) Repeat point a) with a NN with two hidden layers, first one with 32 neurons and
second one with 128 neurons, each followed by ReLU. (2 points)

c) Repeat point a) with a NN with three hidden layers, with 32, 64 and 128 neurons
respectively, each followed by ReLU. (2 points)

d) Repeat point a) with a NN with four hidden layers, with 32, 64, 128 and 128
neurons respectively, each followed by ReLU. (3 points)

```
### check Notebook for more details
[Notebook](02_ImagePredictions_with_NeuralNetworks\Predict_Pixel_Value.ipynb)


## Proj_3: Machine Learning Modelling 

```
Project entailed:

* Become familiar with pandas for handling small datasets
* Use the tf.Estimator and Feature Column API to experiment with feature transformations
* Use visualizations and run experiments to understand the value of feature transformations

```

### check Notebook for more details
[Notebook](03_MachineLearningModelling\Data_MachineLearning_Modelling.ipynb)