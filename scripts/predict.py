# Import libraries
import torch
import json
import argparse
import numpy as np
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt


############# 
# Command line argument parsing
#
# Initialise parser
CLI_parser = argparse.ArgumentParser(description='Please pass positional and optional parameters via CLI to generate prediction as required')
#
# Add positional parameters to parse
CLI_parser.add_argument('image_path', help='File path to image requiring prediction: (e.g. \'flowers/test/55/image_04740.jpg\')')
CLI_parser.add_argument('checkpoint', help='File path (or name) of model checkpoint file: (e.g. \'checkpoint.pth\')')
# Add optional parameters to parse
CLI_parser.add_argument('--device', help='Device for prediction.  Specify \'GPU\' to use GPU.', default='cpu')
CLI_parser.add_argument('--top_k', type=int, help='Top k probabilties and classes to predict', default=5)
CLI_parser.add_argument('--category_names', help='File name for mapping of category classes to names', default='cat_to_name.json')
#
# Parse the arguments
args = CLI_parser.parse_args()
#
##############

# Reassign top_k to default (5) if <=0 is specified
if args.top_k<=0:
    print('You didn\'t specify the number of probabilities and classes to predict.  This has been changed to 5.')
    args.top_k=5

# Execute data prep helper functions
exec(open("helpers_data_prep.py").read())

# Execute modeling helper functions
exec(open("helpers_modeling.py").read())



# Set device to CUDA if available and if specified by user, otherwise use CPU for prediction                       
device = torch.device("cuda:0" if (torch.cuda.is_available()) & (args.device=='GPU') else "cpu")

# Load model checkpoint and assign model and checkpoint dictionary from checkpoint file
model, cp = load_checkpoint(args.checkpoint)

# # Verify correct load of model checkpoint by confirming same accuracy calculation on test data
# # Call function to return a dictionary of datasets and dataloaders from data_dir
# dataset_dict = return_raw_data('flowers')
# # # Print accuracy of loaded model on test set
# check_accuracy_on_test(model, dataset_dict['test_dataloader'])

# Call function to predict top_k highest probabilities and most likely classes
top_k_probs, top_k_classes = predict(args.image_path, model, topk=args.top_k)

# Load category-to-class mapping file and assign to dictionary
cat_to_name = return_cat_class_map(args.category_names)
# Define list of class names for top_k_classes predicted by model
class_labels = [cat_to_name[c] for c in top_k_classes]

# Print top_k class probabilities and class names to screen
print(f'\nPrinting probabilities and class names for top {args.top_k} most likely classes:')
for (rank, (class_name, class_prob)) in enumerate(zip(class_labels, top_k_probs)):
    print(f'{rank+1}. With probability = {round(class_prob,4)}.... Class name is {class_name}') 

print('\n'+'*'*100+f'\n Thank you for using this image classifer to predict the object in {args.image_path}')
print('*'*100+'\n')  
    



