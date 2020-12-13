# import torch
# import torch.nn.functional as F
# from torch import nn
# from torchvision import models

# Define helper functions required for modeling

# Define a new classifier to replace the classifier in a pre-trained model,
# with an arbitrary number of hidden layers and drop out between the layers
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
           Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
    
    # Define forward pass function, which returns log-softmax, applying ReLU activation
    # function and droupout at each hidden layer
    def forward(self, x):
        
        # Loop through hidden layers, applying ReLU activation function and drop out
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        # Generate and return output
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    
 
# Define function to build and return an untrained model, along with criterion and optimizer
# This function allows different hidden layer architectures for the classifier to be easily tested
def return_model_crit_optim(hidden_layers, input_size=25088, output_size=102, model=models.vgg13(pretrained=True), lr=0.001,  drop_p=0.5):
    
    # Freeze parameters of pre-trained model above so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Instantiate feed forward classifier and assign it to classification component of pre-trained model
    classifier = Network(input_size, output_size, hidden_layers, drop_p)
    model.classifier = classifier
    #print(model)

#   Move model to device
    model = model.to(device)

    # Define loss criterion and optimiser
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    
    return model, criterion, optimizer
    

# # Train and view loss and accuracy on validation set

# Define a validation function to be called while training a model
# This validation function will return the loss and accuracy of the model on a validation dataset
def validation(model, validation_dataloader, criterion):
    
    # Initialise test loss and accuracy 
    test_loss = 0
    test_accuracy = 0
    
    # Grab batches of (images, labels) from validation dataloader and forward pass through model
    for (images, labels) in iter(validation_dataloader):
        
        # Move images and labels to device   
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        # Increment test_loss and test_accuracy for batch of validation data
        test_loss += criterion(outputs, labels).item()
        
        # Calculate probabilities for log-softmax output, and correct predictions
        ps = torch.exp(outputs)
        correct_preds = (labels.data == ps.max(dim=1)[1])
        test_accuracy += correct_preds.type(torch.FloatTensor).mean()
        
    return test_loss, test_accuracy


# Define function to train a deep learning model and display the loss and accuracy on a validation test set
# This function will return the trained model
def train_and_validate_model(model, epochs=5, print_every=25):  
    
    # Initialise training_step and running _loss values
    training_step = 0
    running_loss = 0

    for epoch in range (epochs):

        # Set model to training mode
        model.train()

        # Grab batches of (images, labels) from training dataloader
        for (images, labels) in iter(train_dataloader):
            # Move inputs and labels to device
            images, labels = images.to(device), labels.to(device)

            training_step += 1                                  # Increment training step counter
            optimizer.zero_grad()                               # Zero gradients to avoid accumulation of quantities
            outputs = model.forward(images)                     # Forwrard pass through model
            loss = criterion(outputs, labels)                   # Calculate the loss
            loss.backward()                                     # Perform backward pass
            optimizer.step()                                    # Update weights
            running_loss += loss.item()                         # Accumulate running loss with each training step

            # Display loss and accuracy metrics on training and validation datasets
            if training_step % print_every == 0:
                # Set model to evaluation mode, to switch off drop out for evaluation
                model.eval()
                # Turn off gradient calculation to speed up predictions
                with torch.no_grad():
                    val_loss, val_acc = validation(model, valid_dataloader, criterion)
                    print(f'Epoch {epoch+1}/{epochs}: Training loss = {running_loss/print_every}, Validation loss = {val_loss/len(valid_dataloader)}, Validation accuracy = {val_acc/len(valid_dataloader)}')

                running_loss = 0

                # Set model back to training mode
                model.train()
                
    return model
                              
# Define function to calculate and display model accuracy on test dataset
def check_accuracy_on_test(model, testloader):    
    correct_count, all_count = 0, 0
    
    with torch.no_grad():
        for (images, labels) in iter(testloader):
            
            # Move input and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)
            
            # Generate predicted probabilities
            pred_probs = torch.exp(model(images))
            
            # Assign max probability and predicted class
            max_prob, predicted = torch.max(pred_probs.data, 1)
            all_count += labels.size(0)
            correct_count += (predicted == labels).sum().item()

    print(f'Model accuracy on {all_count} test images: {round((100 * correct_count / all_count),2)}%')
    
# Define function to save model check point to folder (optionally) specified by user in CLI
def save_model_checkpoint():
    # Assign class_to_idx mapping as an attribute of the trained model
    trained_model.class_to_idx = train_image_dataset.class_to_idx
    
    # Create a dictionary of paramaters required to rebuild model

    checkpoint = {'input_size':25088,
                  'output_size':102,
                  'hidden_layers':[each.out_features for each in trained_model.classifier.hidden_layers],
                  'lr':args.lr,
                  'drop_rate':trained_model.classifier.dropout.p,
                  'epochs':args.epochs,
                  'arch':'vgg11',
                  'classifier': trained_model.classifier,
                  'state_dict':trained_model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx_map': trained_model.class_to_idx}


    # Save model architecture to checkpoint.pth file in optionally specified folder
    if args.save_dir:
        filepath = args.save_dir+'/checkpoint.pth'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    else:
        filepath = 'checkpoint.pth'
    torch.save(checkpoint, filepath)
    
    print(f'\n\nModel parameters saved in {filepath}')
    

# Define function to load checkpoint file and return previously trained model along with checkpoint parameter dictionary
def load_checkpoint(filepath):
    
    cp = torch.load(filepath)
    
    if cp['arch'] == 'vgg11':
        model = models.vgg11(pretrained = True)
    elif cp['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif cp['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif cp['arch'] == 'vgg19':
        model = models.vgg19(pretrained = True)
    else:
        print('Invalid model')
     
    # Freeze parameters of pre-trained model above so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Assign classifier component of pre-trained model
    model.classifier = cp['classifier']
    
    # Assign previously trained model paramaters
    model.load_state_dict(cp['state_dict'])
    
    # Move model to device
    model = model.to(device)
   
    return model, cp
    
    
# Define function to accept an image path and a model and return top k probabilities and classes
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Pre-process image and convert to a float type tensor on cuda
    np_image = process_image(image_path)
    tensor_image = torch.from_numpy(np_image).to(device, dtype=torch.float)

    # Turn off gradient to speed up calculation
    with torch.no_grad():
        # Ensure model is on cuda
        model = model.to(device)
        
        # Generate predictions
        pred = torch.exp(model.forward(tensor_image.unsqueeze_(0)))
        top_k_probs = pred.topk(k=topk)[0].tolist()[0]
        top_k_indexes = pred.topk(k=topk)[1].tolist()[0]

        # Invert class-to-index dictionary
        index_to_class_map = {v:k for (k,v) in cp['class_to_idx_map'].items()}

        # Define list of top 5 prediction classes
        top_k_classes = [index_to_class_map[ind] for ind in top_k_indexes]

        return top_k_probs, top_k_classes