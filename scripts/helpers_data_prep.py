# Import libraries
# import torch
# import json
# from torchvision import datasets, transforms

# Define helper functions required for processing image data

# Define a function to return a dictionary of transformed datasets and dataloaders using data in data_dir argument
def return_raw_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation and testing sets
    # Augement training data with random transformations
    train_data_transforms = transforms.Compose([
                                                transforms.Resize((224,224)),                # Resize images, as required by pre-trained networks
                                                transforms.RandomHorizontalFlip(),           # Randomly flip images horizontally with default p=0.5
                                                transforms.RandomVerticalFlip(),             # Randomly flip images vertically with default p=0.5
                                                transforms.RandomRotation(degrees=(20,60)),  # Randomly rotate image between 20 and 60 degrees
                                                transforms.ToTensor(),                       # Transform image to a tensor
                                                # Normalise color channels as per ImageNet training images
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])

    # Transform validation data to align with training data, without augmentation
    valid_data_transforms = transforms.Compose([
                                                transforms.Resize((224,224)),                # Resize images, as required by pre-trained networks
                                                transforms.ToTensor(),                       # Transform image to a tensor
                                                # Normalise color channels as per ImageNet training images
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])

    # Transform test data to align with training data, without augmentation
    test_data_transforms = transforms.Compose([
                                               transforms.Resize((224,224)),                # Resize images, as required by pre-trained networks
                                               transforms.ToTensor(),                       # Transform image to a tensor
                                               # Normalise color channels as per ImageNet training images
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])

    # Load the transofmred datasets and store insert in dataset_dict
    dataset_dict = {}
    dataset_dict['train_image_dataset'] = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    dataset_dict['valid_image_dataset'] = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)
    dataset_dict['test_image_dataset'] = datasets.ImageFolder(test_dir, transform=test_data_transforms)

    # Define dataloaders and add to dataset_dict
    dataset_dict['train_dataloader'] = torch.utils.data.DataLoader(dataset_dict['train_image_dataset'], batch_size=64, shuffle=True)
    dataset_dict['valid_dataloader'] = torch.utils.data.DataLoader(dataset_dict['valid_image_dataset'], batch_size=32)
    dataset_dict['test_dataloader'] = torch.utils.data.DataLoader(dataset_dict['test_image_dataset'], batch_size=32)
    
    return dataset_dict

# Define function to return category-to-class name mapping
def return_cat_class_map(category_class_filename):
    with open(category_class_filename, 'r') as f:
        return json.load(f)

# Define function to return resized image such that shortest side is 256 pixels
# and the image's aspect ratio is retained
def return_resized_img(img, min_dim=256):

    # Get image dimensions
    img_width, img_height = img.size

    # Calculate scaled heights and widths    
    if img_height <= img_width:
        scaling_factor = min_dim / img_height
        scaled_height = int(scaling_factor * img_height)
        scaled_width = int(scaling_factor * img_width) 
    else: 
        scaling_factor = min_dim / img_width
        scaled_height = int(scaling_factor * img_height)
        scaled_width = int(scaling_factor * img_width)

    # Return resized image, with scaled width and height    
    return img.resize((scaled_width, scaled_height))


# Define function to return coordinates of box required for cropping
def find_center_crop_coords(img, new_width=224, new_height=224):

    # Get image dimensions
    img_width, img_height = img.size   

    # Calculate borders
    left = (img_width - new_width)/2
    top = (img_height - new_height)/2
    right = (img_width + new_width)/2
    bottom = (img_height + new_height)/2

    # Return coordinates of cropped box borders
    return left, top, right, bottom    

# Define function to accept an image_path and return the image as a NumPy array in format
# required by model.  Pre-processing will be conducted on image to ensure it's
# consistent with training data images
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)

    # Create resized and cropped versions of image
    img_resized = return_resized_img(img)
    img_cropped = img_resized.crop((find_center_crop_coords(img_resized)))

    # Convert cropped image to a numpy array
    np_image = np.array(img_cropped)

    # Scale values between 0 and 1
    np_image = np_image / 255

    # Normalise values in same way as training data
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std

    np_image = np_image.transpose()
    
    return np_image