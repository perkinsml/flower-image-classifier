# Table of Contents

1. [Background](#background)
1. [Project Overview](#project-overview)
1. [Using the command line application](#using-the-command-line-application)
1. [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Executing the program](#executing-the-program)
    * [File Descriptions](#file-descriptions)
1. [Machine Learning considerations](#machine-learning-considerations)
1. [Author](#author)
1. [License](#license)
1. [Acknowledgements](#acknowledgements)
1. [Web App Screenshots](#web-app-screenshots)



# Background
ML algorithms are being incorporated into more and more every day applications, such as mobile phones and watches.  Enabling image classification from a smart phone app (for example), typically requires training a deep learning Neural Network on hundreds of thousands of images to build an accurate classifier that can be used as part of the application's architecture.  Software Developers are then required to deploy this model in such a way that users can efficiently leverage these trained models to perform inference/prediction/image classification (for example) from their everyday devices.  

# Project Overview
This project uses PyTorch to train an image classifier to recognise different species of flowers.  Such a classifier could be used within an app on a mobile phone to inform a user of the name of the flower their camera is looking at.  

The dataset used to build this classifier is the [102 Flower Category Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  As the name suggests, this dataset consists of 102 different flower categories (which are commonly occurring in the United Kingdom), with each flower class consisting of between 40 and 258 images.  Some of these flower categories have large variations within the category and several categories are very similar.  Note, this flower image data set is too large to upload to GitHub, so is not included in this repo.  The data can be accessed [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), if required.  Examples of the flower images are displayed below:  

![flower images example](https://github.com/perkinsml/flower-image-classifier/blob/master/images/flowers_example.png)

This projects consists of 2 key parts:

1. **Development of an image classifier on the *102 Flower Catagory Dataset* in a Jupyter Notebook**: Leveraging Transfer Learning, this deep learning neural network consists of a pre-trained neural network (trained on the ImageNet dataset) to detect and extract features from the images, which then feeds forward into a custom built and trained classification layer. This image classifier is  defined, trained and evaluated within this Jupyter Notebook.
1. **Development of a command line application**: the code developed in the previous part of this project is converted into an application (consisting of several scripts) that can be run from the command line and used to train an image classifier on any image dataset, and/or use a trained image classifier for image category prediction. Please refer to the instructions below regarding how to use this command line application.


# Installation
Clone this GitHub repository:

```
git clone https://github.com/perkinsml/flower-image-classifier.git
```


## Dependencies
Package versions included:
* Python 3.5+ (I used Python 3.6.3)
* Torch: 0.4.0
* TorchVision: 0.2.1
* NumPy: 1.12.1
* Matplotlib: 2.1.0
* Pillow: 5.2.1



## Using the command line application
Run the commands in the project's **scripts** folder described below to use the command line application.

1. Train a deep learning neural network on an image data set and save the model as a checkpoint, by executing the <code>train.py</code> script from the command line in the **scripts** folder, as per the instructions below.
<p>  Basic usage: <code>python train.py data_directory</code>, where *data_directory* is the parent directory containing the images used for model training, validation and testing.  An example folder structure for the training, validation and testing image data is displayed below:
<pre><code>
├── data_directory                          # Parent folder of image dataset
       ├── train                            # Parent folder of images used for model training
       ├── valid                            # Parent folder of images used for model validation
       └── test                             # Parent folder of images used for model testing
</code></pre>
<p> The training loss, validation loss and validation accuracy are printed to screen as the model trains - see example below:
![metrics display example](https://github.com/perkinsml/flower-image-classifier/blob/master/images/metrics_display_example.png)

<p>After training is complete, the accuracy on the (hold-out) test set will be calculated and printed to screen.  The model checkpoint will then be saved to the same folder as the <code>train.py</code> script.</p>
<p>The default parameters for the <code>train.py</code> script can be modified by specifying one or more of the optional parameters listed below when executing the <code>train.py</code> script from the command line.

  <ul>
      <li>TorchVision's pre-trained model architecture used for image feature detection layers.  Note, only VGG architectures are permitted (default = VGG11): <code>python train.py data_dir --arch vgg13</code></li>
      <li>Hidden layer architecture for image classification layer (default = [1024, 512, 256]): <code>python train.py data_dir --hidden_layers 2048 1024 512</code></li>
      <li>Number of training epochs (default = 20): : <code>python train.py data_dir --epochs 50</code></li>
      <li>Learning rate for the classification layer (default = 0.001): <code>python train.py data_dir --lr 0.05</code></li>
      <li>Drop-out probability for nodes in classification layer (default = 0.5): <code>python train.py data_dir --drop_out 0.35</code></li>
      <li>Device for model training (default = cpu): <code>python train.py data_dir --device GPU</code></li>
      <li>Folder to save model checkpoint to (default = same folder as the <code>train.py</code> script): <code>python train.py data_dir --save_dir save_directory</code></li>  
  </ul>

<br> For example, to train an image classifier using TorchVision's pre-trained VGG13 model architecture feeding into a classification layer consisting of 4 hidden layers with 256, 128, 64 and 32 nodes respectively, on a GPU for 50 epochs with a learning rate of 0.005 and drop out probability of 0.35, I'd execute the command below at command line.  The command below would use the images within the animals folder (as per the directory structure example above) and save the model checkpoint to the models folder.


    ```
    python train.py animals --arch vgg13 --hidden_layers 256 128 64 32 --epochs 50 --lr 0.005 --drop_out 0.35 --device GPU --save_dir models
    ```

1. To use a pre-trained image classifier for inference:

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
