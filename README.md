# Table of Contents

1. [Background](#background)
1. [Project Overview](#project-overview)
1. [Data](#data)
1. [File Descriptions](#file-descriptions)
1. [Flower image classification results](#flower-dataset-image-classification-results)
1. [Command Line Application](#command-line-application)
    * [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Using the command line application](#using-the-command-line-application)
1. [Author](#author)
1. [License](#license)
1. [Acknowledgements](#acknowledgements)


# Background
ML algorithms are being incorporated into more and more everyday applications, such as mobile phones and watches.  

Enabling image classification from a smart phone app (for example), typically requires training a deep learning Neural Network on thousands of images to build an accurate classifier that can be used as part of the application's architecture.  Software Developers are then required to deploy this model in such a way that users can efficiently leverage these trained models to perform inference/prediction/image classification (for example) from their everyday devices.  

# Project Overview
This project uses PyTorch to train an image classifier to recognise different species of flowers.  Such a classifier could be used within an app on a mobile phone to inform a user of the name of the flower their camera is looking at.  

This projects consists of 2 key parts:

1. **Development of an image classifier trained with the [102 Flower Category Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) in a Jupyter Notebook**: Leveraging Transfer Learning, this deep learning neural network consists of a pre-trained neural network (trained on the ImageNet dataset) to detect and extract features from the images, which then feeds forward into a custom built and trained classification layer. This image classifier is  defined, trained and evaluated within a Jupyter Notebook.
1. **Development of a command line application**: the code developed in the previous part of this project is converted into an application (consisting of several Python scripts) that can be run from the command line and used to train an image classifier on any image dataset, and/or use a trained image classifier for image category prediction. Please refer to the [Command Line Application section](#command-line-application) for  instructions regarding how to use this command line application.

# Data
The dataset used to build the image classifier is the [102 Flower Category Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  As the name suggests, this dataset consists of 102 different flower categories (which are commonly occurring in the United Kingdom), with each flower class consisting of between 40 and 258 images.  Some of these flower categories have large variations within the category and several categories are very similar.  The data consists of 6,552 training images, 818 validation images and 819 (hold-out) test images.  

Note, this flower image data set is too large to upload to GitHub, so is not included in this repo.  The data can be accessed [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), if required.  

Examples of the flower images are displayed below:  

![flower images example](https://github.com/perkinsml/flower-image-classifier/blob/master/images/flowers_example.png)

# File Descriptions

<pre><code>
|
├── image_classifier_project.ipynb       # Jupyter Notebook with the code required to:
|                                        # - load and preprocess the image data
|                                        # - build and train a classifier leveraging transfer learning techniques
|                                        # - use the trained classifier to predict the image category
|
├── image_classifier_project.html        # The HTML version of the image_classifier_project.ipynb
|
├── scripts                              # A folder of Python scripts for the command line application
│   ├── train.py                         # A script which can be executed from the command line (see instructions below) to train a new network on a dataset and save the model as a checkpoint
│   ├── predict.py                       # A script which can be executed from the command line (see instructions below) to use a pre-trained network to predict the class of a specified image
│   ├── helpers_data_prep.py             # A script containing helper functions required for image pre-processing (including transforms and augmentations)
|   ├── helpers_modeling.py              # A script containing helper functions for model definition, training, validation, testing and inference
|   └── cat_to_name.json                 # A JSON file for mapping category ids to category names
|
├──images                                # A folder of images used on this page
|
└── README.md
</code></pre>


# Flower Dataset Image Classification results
As can be seen in image_classifier_project.ipynb, an accuracy of 0.74 is achieved on a hold-out test set of 819 flower images.  The neural network architecture consists of a pre-trained VGG11 model feeding into a classification model with 3 hidden layers - with 1024, 512 and 256 nodes respectively.  A learning rate of 0.001 and a drop out probability of 0.35 is used in model training.

The model is trained with 6,552 images that have been resized and normalised (as required by the pre-trained networks), and randomly augmented.  Having achieved an accuracy of 0.75 on the 818 images in the validation set, the model generalises well to the test set.

Below is an example of the inference results for the test images displayed at the top of this page.
<p>
![inference probability example](https://github.com/perkinsml/flower-image-classifier/blob/master/images/inference_probability_example.png)



# Command Line Application

## Installation
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
<p>The command line application provides the options to train a new neural network, or use an existing neural network checkpoint for the prediction of an image's category.

<p>Run the commands described below in the project's <b>scripts</b> folder to use the command line application.

<ol>
<li>
<p><b>Train a deep learning neural network on an image data set and save the model as a checkpoint</b>, by executing the <code>train.py</code> script from the command line in the <b>scripts</b> folder, as per the instructions below.

<p>Basic usage: <code>python train.py data_directory</code>, where <i>data_directory</i> is the parent directory containing the images used for model training, validation and testing.  Note, <i>data_directory</i> is a mandatory argument when running the <code>train.py</code> script.  An example folder structure for the training, validation and testing image data is displayed below:
<pre><code>
├── data_directory                       # Parent folder of image dataset
    ├── train                            # Parent folder of images used for model training
    ├── valid                            # Parent folder of images used for model validation
    └── test                             # Parent folder of images used for model testing
</code></pre>

<p> The training loss, validation loss and validation accuracy are printed to screen as the model trains - see example below:</p>

![metrics display example](https://github.com/perkinsml/flower-image-classifier/blob/master/images/metrics_display_example.png)

<p>After training is complete, the accuracy on the (hold-out) test set will be calculated and printed to screen.  The model checkpoint will then be saved to the same folder as the <code>train.py</code> script.
<p>The default parameters for the <code>train.py</code> script can be modified by specifying one or more of the optional parameters listed below when executing the <code>train.py</code> script from the command line.

  <ul>
      <li>TorchVision's pre-trained model architecture used for image feature detection layers.  Note, only VGG architectures are supported (default = VGG11): <code>python train.py data_dir --arch vgg13</code></li>
      <li>Hidden layer architecture for image classification layer (default = [1024, 512, 256]): <code>python train.py data_dir --hidden_layers 2048 1024 512</code></li>
      <li>Number of training epochs (default = 20): : <code>python train.py data_dir --epochs 50</code></li>
      <li>Learning rate for the classification layer (default = 0.001): <code>python train.py data_dir --lr 0.05</code></li>
      <li>Drop-out probability for nodes in classification layer (default = 0.5): <code>python train.py data_dir --drop_out 0.35</code></li>
      <li>Device for model training (default = cpu): <code>python train.py data_dir --device GPU</code></li>
      <li>Folder to save model checkpoint to (default = same folder as the <code>train.py</code> script): <code>python train.py data_dir --save_dir save_directory</code></li>  
  </ul>

<br> For example, training an image classifier using TorchVision's pre-trained VGG13 model architecture feeding into a classification layer consisting of 4 hidden layers with 256, 128, 64 and 32 nodes respectively, on a GPU for 50 epochs with a learning rate of 0.005 and drop out probability of 0.35, can be executed from the command line with the command below.  This command would use the images within the <i>animals</i> folder (as per the directory structure example above) and save the model checkpoint to the <i>models</i> folder.

```
python train.py animals --arch vgg13 --hidden_layers 256 128 64 32 --epochs 50 --lr 0.005 --drop_out 0.35 --device GPU --save_dir models
```
</li>
<li>
<p><b>Predict the category of an image using a pre-trained neural network</b>, by executing the <code>predict.py</code> script from the command line in the <b>scripts</b> folder, as per the instructions below.

<p>Basic usage: <code>python predict.py path_to_image path_to_checkpoint</code>, where <i>path_to_image</i> is the file path to the image for inference and <i>path_to_checkpoint</i> is the file path to the pre-trained image classifier to be used for inference.  Note, <i>path_to_image</i> and <i>path_to_checkpoint</i> are both mandatory arguments when running the <code>predict.py</code> script.

<p>By default, the top 5 most likely image classes and their probabilities will be printed to screen, as illustrated by the example below.

![inference display example](https://github.com/perkinsml/flower-image-classifier/blob/master/images/inference_display_example.png)

<p>The default parameters for the <code>predict.py</code> script can be modified by specifying one or more of the optional parameters listed below when executing the <code>predict.py</code> script from the command line.

  <ul>
      <li>Device for model training (default = cpu): <code>python predict.py path_to_image path_to_checkpoint --device GPU</code></li>
      <li>Top k probabilities and classes to predict (default = 5): <code>python predict.py path_to_image path_to_checkpoint --top_k 10</code></li>
      <li>File name for mapping of category classes to names (default = 'cat_to_name.json'): <code>python predict.py path_to_image path_to_checkpoint --category_names mapping.json</code></li>
  </ul>



# Author
[Matthew Perkins](https://github.com/perkinsml)

# License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Acknowledgements
* [Udacity](https://www.udacity.com/) for designing the Project
* [Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/) for open sourcing the labelled 102 Flower Category Dataset
