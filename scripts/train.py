# Import libraries
import torch
import json
import argparse
import torch.nn.functional as F
import numpy as np
import os
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
from matplotlib import pyplot as plt
# from os import path, mkdir


############# 
# Command line argument parsing
#
# Initialise parser
CLI_parser = argparse.ArgumentParser(description='Please pass positional and optional parameters to train an image classifier as required')
#
# Add positional parameters to parse
CLI_parser.add_argument('data_directory', help='Parent folder of model image inputs (e.g. \'flowers\')')
# Add optional parameters to parse
CLI_parser.add_argument('--hidden_layers', nargs='*', type=int, help='Hidden layers for Classifier e.g. 1024 512 256', default=[1024, 512, 256])
CLI_parser.add_argument('--arch', help='Name of pre-trained VGG Torchvision model', default='vgg11')
CLI_parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
CLI_parser.add_argument('--lr', type=float, help='Learning rate for the Classifier', default=0.001)
CLI_parser.add_argument('--drop_out', type=float, help='Drop out probability', default=0.5)
CLI_parser.add_argument('--device', help='Device for training.  Specify \'GPU\' to train model on GPU.', default='cpu')
CLI_parser.add_argument('--save_dir', help='Folder to save model checkpoint')
#
# Parse the arguments
args = CLI_parser.parse_args()
#
##############

# Execute data prep helper functions
exec(open("helpers_data_prep.py").read())

# Execute modeling helper functions
exec(open("helpers_modeling.py").read())

# Assign parent directory for training, validation and test data
data_dir = args.data_directory
# Call function to return a dictionary of datasets and dataloaders from data_dir
dataset_dict = return_raw_data(data_dir)
# Unpack dataloaders from dictionary
train_image_dataset = dataset_dict['train_image_dataset']
valid_image_dataset = dataset_dict['valid_image_dataset']
test_image_dataset = dataset_dict['test_image_dataset']
train_dataloader = dataset_dict['train_dataloader']
valid_dataloader = dataset_dict['valid_dataloader']
test_dataloader = dataset_dict['test_dataloader']

# Display train, validation and test dataset properties
# print(train_image_dataset)
# print(valid_image_dataset)
# print(test_image_dataset)

# Assign model with a classifier consisting of hidden_layers specified in CLI as an optional 
# paramater (default=[1024, 512, 256]), criterion and optimizer 
if args.arch == 'vgg11':
    tv_model = models.vgg11(pretrained=True)
elif args.arch =='vgg13':
    tv_model = models.vgg13(pretrained=True)
elif args.arch =='vgg16':
    tv_model = models.vgg16(pretrained=True)
elif args.arch =='vgg19':
    tv_model = models.vgg19(pretrained=True)
else:
    tv_model = models.vgg11(pretrained=True) 
    print('Unsupported architecture specified, using pre-trained model: VGG11')
                                     

# Set device to CUDA if available and if specified by user, otherwise CPU                        
device = torch.device("cuda:0" if (torch.cuda.is_available()) & (args.device=='GPU') else "cpu")
 
# Define model, criterion and optimizer       
model, criterion, optimizer = return_model_crit_optim(args.hidden_layers, model=tv_model, lr=args.lr, drop_p=args.drop_out)
# Display model architecture
print(model)

# Train model with optionally specified epochs
with active_session():
    trained_model = train_and_validate_model(model, epochs=args.epochs)
    

# Validate trained model on test dataset 
print('\n\n'+'*'*100)
print('NOW TESTING TRAINED MODEL ON TEST DATA SET...')
# Set model to evalution mode to switch off drop out and display accuracy on test data
trained_model.eval()
check_accuracy_on_test(trained_model, test_dataloader)
print('*'*100)

# Save train model parameters as a checkpoint
save_model_checkpoint()


    








