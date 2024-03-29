# Traffic Signs Image Classification

## Table of Content
  * [Problem Statement](#Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Model Building and Tuning](#Model-Building-and-Tuning)
  * [Used Techniques](#Other-Used-Techniques)
  * [Demo](#demo)
  * [Run project in your local machine](#Run-the-project-in-your-local-machine)
  * [Bug / Feature Request](#bug---feature-request)


## Business Problem Statement
Traffic sign classification is the process of automatically recognizing traffic signs along the road, including speed limit signs, yield signs, merge signs, etc. Being able to automatically recognize traffic signs enables us to build “smarter cars”. Self-driving cars need traffic sign recognition in order to properly parse and understand the roadway. Similarly, “driver alert” systems inside cars need to understand the roadway around them to help aid and protect drivers.
 Traffic sign recognition is just one of the problems that computer vision and deep learning can solve. In this case study, we have a dataset which includes images of traffic signs (43 different traffic signs) and the goal is to train a Deep Neural Network to classify them.
## Data
Data Source : Private Data Source.

## Used Libraries and Resources
**Python Version** : 3.6

**Libraries** : tensorflow, keras, pandas, numpy, matplotlib, seaborn, flask, json, pickle

**References** : https://towardsdatascience.com/, https://machinelearningmastery.com/


## Data Preprocessing
These are the data-preprocessing techniques that I used to preprare the dataset for training :

* Shuffle the dataset in order to have a better learning from the model.
* To simplify the problem we will convert images from RGB (red-green-blue) to gray in order to decrease the image dimension and as a result the training time. For each image pixel we have three values RGB. The sum of these three values will be calculated to convert the image into gray scale image.
* Feature scaling since Neural Networks perform better when data is scaled. Image pixels range from 0 to 255. Division by 255 leads to values that lie between 0 and 1.


## Model Building and Tuning

* I trained a Convolutional Neural Networks using LeNet architechture which seems to overfit to the training set. 
* To avoid overfitting, I used to the Early Stopping Callback technique. If the accuracy on the training set is greater than 0.98, the training stops. This model got a lower accuracy but also lower loss on test set.
* Hyperparameter Tuning is done using RandomSearch of KerasTuner. Hypyerparameters are randomly selected and tuned. The tuned model got almost the same accuracy as the default model and a lower loss on test set.
* Data Augmentation is used to have a better performance on new images. We get lower accuracy on this particular test set because its distribution is very similar to the training data distribution. Images in test set are not in different shapes,rotated, flipped etc. But when it comes to new images it might perform better than the model that was trained without using data augmentation.
* I applied transfer Learning using the pre-trained model VGG16 which was trained to predict images of 1000 different classes.
* I evaluated each ML model using training score, cross validation score, test score to get a better understanding about the model performances. The best model is selected using the test and validation score.
* The best model I got out of all models is the Tuned LeNet.

| Model Name                 | Test Accuracy            |    Test Loss                | CV Accuracy            |  CV Loss          |   Training Accuracy  | Training Loss  |
|:--------------------------:|:------------------------:|:---------------------------:|:----------------------:|:-----------------:|:--------------------:|:--------------:|
|LeNet NN                    |     0.941                |     0.818                   |         0.957          |     0.467         |         0.998        |   0.009        | 
|LeNet NN with Early Stopping|     0.907                |     0.526                   |         0.938          |     0.256         |         0.987        |   0.048        | 
|Tuned LeNet                 |     0.935                |     0.514                   |            0.960       |     0.253         |         0.999        |   0.003        | 
|LeNet with Data Augmentation|     0.820                |     0.836                   |         -              |   -               |        -             |   -            | 
|VGG16 - Transfer Learning   |     0.8514               |     0.8861                  |         0.8105         |     0.7833        |         0.7800       |   0.7800       |     


## Other Used Techniques

* Built a client facing API (web application) using Flask.

## Demo

This is how the web application looks like : 


![alt text](https://github.com/Lori10/Traffic-Signs-ImageClassification/blob/main/img.PNG "Image")



## Run the project in your local machine 

1. Clone the repository
2. Open the project directory in PyCharm  and create a Python Interpreter using Conda Environment : Settings - Project - Python Interpreter - Add - Conda Environment - Select Python Version 3.6 - 
3. Run the following command in the terminal to install the required packages and libraries : pip install -r requirements.txt
4. Run the file app.py by clicking Run and open the API that shows up in the bottom of terminal.



## Bug / Feature Request
If you find a bug, kindly open an issue.
