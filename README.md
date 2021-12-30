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
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


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
* To avoid overfitting, I used to the Early Stopping Callback technique. If the accuracy on the training set is greater than 0.98, the training stops.
* Hyperparameter Tuning is done using RandomSearch of KerasTuner. Hypyerparameters are randomly selected and tuned.
* Data Augmentation is used to have a better performance on new images.
* Transfer Learning technique is used to check if a pre-trained model can outperform the model which we created.
* I evaluated each ML model using training score, cross validation score, test score to get a better understanding about the model performances. The best model is selected using the test score.
* The best model I got out of all models is Random Forest with an accuracy of .8793.

| Model Name        | Deafult Model Test Score |Default Model Training Score | Default Model CV Score | Tuned Model Test Score | Tuned Model Training Score | Tuned Model CV Score | 
|:-----------------:|:------------------------:|:---------------------------:|:----------------------:|:----------------------:|:--------------------------:|:---------------------:|
|Linear Regression  |     0.7891               |     0.7833                  |         0.7800         |      0.7891            |           0.7833           |     0.7800             |
|Random Forest      |     0.8794               |     0.9700                  |         0.8758         |      0.8793            |           0.7833           |     0.8792            |
|KNN                |     0.8514               |     0.8861                  |         0.8105         |      0.8504            |           0.9824           |  0.8248              |


## Other Used Techniques

* Object oriented programming is used to build this project in order to create modular and flexible code.
* Built a client facing API (web application) using Flask.
* A retraining approach is implemented using Flask framework.
* Using Logging every information about data cleaning und model training HISTORY (since we may train the model many times using retraining approach)  is stored is some txt files and csv files for example : the amount of missing values for each feature, the amount of records removed after dropping the missing values and outliers, the amount of at least frequent categories labeled with 'other' during encoding, the dropped constant features, highly correlated independent features, which features are dropping during handling multicolleniarity, best selected features, model accuracies and errors etc.

## Demo

This is how the web application looks like : 


![alt text](https://github.com/Lori10/Banglore-House-Price-Prediction/blob/master/Project%20Code%20Pycharm/demo_image.jpg "Image")



## Run the project in your local machine 

1. Clone the repository
2. Open the project directory (PyCharm Project Code folder) in PyCharm  and create a Python Interpreter using Conda Environment : Settings - Project : Project Code Pycharm - Python Interpreter - Add - Conda Environment - Select Python Version 3.6 - 
3. Run the following command in the terminal to install the required packages and libraries : pip install -r requirements.txt
4. Run the file app.py by clicking Run and open the API that shows up in the bottom of terminal.


## Directory Tree 
```
 ├── Project Code PyCharm├── static 
                             ├── css
                                 ├── styles.css
                         ├── templates
                         │   ├── home.html
                         ├── File_Operation
                             ├── FileOperation.py
                         ├── Functions
                             ├── functions.py
                         ├── Logs
                             ├── DataPreprocessing_Log.txt
                             ├── ModelTraining_Log.txt
                             ├── Prediction_Log.txt
                         ├── ModelTraining
                             ├── trainingModel.py
                         ├── Training_FileFromDB
                             ├── dataset.csv
                         ├── application_logging
                             ├── logger.py
                         ├── best_model_finder
                             ├── modelTuning.py
                         ├── data_ingestion
                             ├── data_loader.py
                         ├── data_preprocessing
                             ├── preprocessing.py
                         ├── models
                             ├── RandomForestRegressor
                                 ├── RandomForestRegressor.sav
                         ├── app.py
                         ├── encoded_features.json
                         ├── model_infos.csv
                         ├── multicolleniarity_heatmap.jpg
                         ├── nan_values.csv
                         ├── Training Infos.ipynb
                         ├── requirements.txt
```



## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Lori10/Banglore-House-Price-Prediction/issues) here by including your search query and the expected result

## Future Scope

* Use other ML Estimators
* Try other feature engineering approaches to get a possible higher model performance
* Optimize Flask app.py Front End
