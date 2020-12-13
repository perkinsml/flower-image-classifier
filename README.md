# Table of Contents

1. [Background](#background)
1. [Project Overview](#project-overview)
1. [Using the web application](#using-the-web-application)
1. [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Executing the program](#executing-the-program)
    * [File Descriptions](#file-descriptions)
1. [Machine Learning considerations](#machine-learning-considerations)
1. [Author](#author)
1. [License](#license)
1. [Acknowledgements](#acknowledgements)
1. [Web App Screenshots](#web-app-screenshots)

![web app header](https://github.com/perkinsml/disaster_response_pipeline/blob/master/images/web_app_header.png)

# Background
ML algorithms are being incorporated into more and more every day applications, such as mobile phones and watches.  Enabling image classification from a smart phone app for example, typically requires training a deep learning Neural Network on hundreds of thousands of images to build an accurate classifier that can be used as part of the application's architecture.  Software Developers are then required to deploy this model in such a way that users can efficiently leverage these trained models to perform inference/predictions/image classification (for example) from their everyday devices.  

# Project Overview
This project uses PyTorch to train an image classifier to recognise different species of flowers.  Such a classifier could be used within an app on a mobile phone to inform a user of the name of the flower their camera is looking at.  

The dataset used to build this classifier is the [102 Flower Category Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  As the name suggests, this dataset consists of 102 different flower categories (which are commonly occurring in the United Kingdom), with each flower class consisting of between 40 and 258 images.  Some of these categories have large variations within the category and several categories are very similar.  Note, the flower image data set is too large to upload to GitHub, so is not included in this repo.  The data can be accessed [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), if required.  Examples of these images are displayed below:  

![flower images example](https://github.com/perkinsml/flower-image-classifier/blob/master/images/flowers_example.png)

This projects consists of 2 key parts:

1. **Development of the image classifier**: an image classifier is first developed, trained and evaluated in a Jupyter Notebook.  Leveraging Transfer Learning, this deep learning neural network consists of a pre-trained neural network (trained on the ImageNet dataset) to detect and extract features from the images, which then feeds forward into a custom built and trained classification layer. This classification layer is defined, trained and evaluated within this notebook.
1. **Building a command line application**: the code developed in the previous part of this project is converted into an application (consisting of several scripts) that others can run from the command line.  The model checkpoint saved from the first part of this project is used to test this application.  Please refer to the instructions below regarding how to use this application from the command line.

# Using the web application
The **Disaster Response Message Classifier web application is live** and can be accessed [here](https://dismsgclf.herokuapp.com/).  No installations are required to use the web app.

# Installation
Clone this GitHub repository:

```
git clone https://github.com/perkinsml/disaster_response_pipeline.git
```

You'll need to install the dr_utils package included in the repository by typing the command below in the root directory.  

```
pip install .
```

The dr_utils package includes a custom word tokenise function and a model scorer function - both of which are required to run the ML pipeline.  Given the class imbalance of the dataset and the priority of recall in this scenario, a custom f-beta scorer (with beta=2) was used to evaluate the model during grid search.  Please refer to the *ML Pipeline Preparation.ipynb* notebook for more detail.

## Dependencies
A list of dependencies is included in the requirements.txt file in this repository.
* Python 3.5+ (I used Python 3.7.6)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
* Natural Language Process Libraries: NLTK
* SQLite Database Libraries: SQLAlchemy
* Web App and Data Visualization: Flask, Plotly
* Custom functions: dr-utils package (refer to installation instructions above)



## Executing the command line application
Run the following commands in the project's *scripts* directory to train set up the database and model:
1. Train a deep learning neural network on a data set and save the model as a checkpoint by executing the <code>train.py</code> script from the command line, as per the instructions below.  Note,

    <ul>
    <li>Basic usage: <code>python train.py data_directory</code>, where data_dir is the parent directory containing train, val and test image inputs (e.g. \'flowers\')</li>
    <li>Prints out training loss, validation loss, and validation accuracy as the network trains</li>
    <li>Options parameters include:<ul>
    <li>Directory to save model checkpoint: <code>python train.py data_dir --save_dir save_directory</code></li>  
    <li>Name of pre-trained VGG Torchvision model (default='vgg11'): <code>python train.py data_dir --arch 'vgg13'</code></li>
    <li>Hidden layer architecture for Classifier (default=[1024, 512, 256]): <code>python train.py data_dir --hidden_layers '1024 512 256'</code></li>
    <li>Number of training epochs (default=20): <code>python train.py data_dir --epochs '50'</code></li>
    <li>Learning rate for Classifier (default=0.001): <code>python train.py data_dir --lr '0.05'</code></li>
    <li>Drop out probability for Classifier (default=0.5): <code>python train.py data_dir --drop_out 0.35</code></li>
    <li>Use GPU for training (default='cpu'): <code>python train.py data_dir --gpu</code></li>



    <li>Set hyperparameters: <code>python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20</code></li>

    </ul>


    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```

1. To run the ML pipeline that trains and saves the classifier:

   ```
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

Re-running the ETL and ML pipelines is not necessary to start the web app.  

After following the installation instructions above, you can simply execute the following commands to run the web app., regardless of whether or not you choose to re-run the ETL and ML pipelines:
1. From the app directory:
    ```
    export FLASK_APP=run.py
    python -m flask run
    ```
1. Go to the link displayed in the command line.  For me, this is: http://127.0.0.1:5000/


## File Descriptions

<pre><code>
|
├── image_classifier_project.ipynb       # Jupyter Notebook with the code required to:
|                                        # - load and preprocess the image data
|                                        # - build and train a classifier leveraging transfer learning techniques
|                                        # - use the trained classifier to predict image content
|
├── image_classifier_project.html        # The HTML version of the image_classifier_project.ipynb
|
├── checkpoint.pth                       # The model checkpoint trained, evaluated and saved within the image_classifier_project.ipynb



├── scripts
│   ├── train.py                         # A Python script which can be executed from the command line (see instructions above) to train a new network on a dataset and save the model as a checkpoint
│   ├── predict.py                       # A Python script which uses a trained network to predict the class for an input image
│   ├── helpers_data_prep.py             # A Python script containing helper functions required for image pre-processing (including transforms and augmentations)
|   └── helpers_modeling.py              # A Python script containing helper functions for model definition, training, validation, testing and inference
|
├── models
│   ├── train_classifier.py              # ML pipeline script
|   ├── classifier.pkl                   # Pickled classification model
|   └── ML Pipeline Preparation.ipynb    # Notebook demonstrating ML pipeline build, train and test
|
├── dr_utils
|   └── custom_functions.py              # Custom functions used by  classification model
|
├── app
│   ├── run.py                           # Flask file that runs the app
│   └── templates
│       ├── master.html                  # Main page of web app
│       └── go.html                      # Classification results page of web app
│
├── images                               # A folder of screen shots used on this page
|
├── requirements.txt                     # A list of required libraries and their versions
|
└── README.md
</code></pre>

# Machine Learning considerations
The dataset includes 36 message categories - one of which ('child_alone') is not relevant to any messages.  After testing a range of classification algorithms, I found the LinearSVC model to achieve the best results.  The solver for this algorithm requires at least 2 classes in the data, so the 'child_alone' category was dropped from the data.

As shown in the web app and the *ML Pipeline Preparation.ipynb* notebook, the dataset is imbalanced and just 3 of the remaining 35 categories have more than 20% of messages assigned to them. Given this class imbalance, accuracy was not the most robust metric for evaluating the model performance.  Given the use case and hence the importance of recall (i.e. the need to identify messages relevant for each category) in this situation, I elected to use a the mean F2 score across all 35 categories to evaluate model performance.   

# Author
[Matthew Perkins](https://github.com/perkinsml)

# License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Acknowledgements
* [Udacity](https://www.udacity.com/) for designing the Project
* [Figure8 (now known as Appen)](https://appen.com/) for collating and labelling the dataset



# Web App Screenshots
Below is an example of the categorisation results displayed by the web app for the message:

>A massive fire has broken out after the storm. Homes are destroyed<br> and people have been left homeless.  We need doctors and clothing.

![results summary image](https://github.com/perkinsml/disaster_response_pipeline/blob/master/images/web_app_results_example.png)

The main page of the web app displays some visualisations of the message data provided by Figure Eight

![data charts image](https://github.com/perkinsml/disaster_response_pipeline/blob/master/images/data_overview.png)
