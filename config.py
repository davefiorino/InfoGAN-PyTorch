# Dictionary storing network parameters.
params = {
    'batch_size': 256, # Batch size.
    'num_epochs': 100, # Number of epochs to train for. 
    'learning_rate_D': 2e-4, # Learning rate for discriminator
    'learning_rate_G': 2e-4,# learning rate for generator 
    'beta1': 0.5, # Momentum parameter
    'beta2': 0.999, # RMSprop parameter
    'save_epoch' : 20, # After how many epochs to save checkpoints
    'dataset' : 'CelebA', # Choose from {MNIST, SVHN, CelebA, FashionMNIST, ChestXRay}
    'print_model_description' : True, # Print the description of the models during training and generation
    'image_log_epoch': 10, # Save a generated image every X epochs
    'use_dropout': False # Use dropout layers in the first half epochs
}

def init():
    global currentEpoch
    currentEpoch = 0