# Dictionary storing network parameters.
params = {
    'batch_size': 64, # Batch size.
    'num_epochs': 40, # Number of epochs to train for. 
    'learning_rate_D': 2e-4, # Learning rate for discriminator
    'learning_rate_G': 2e-4,# learning rate for generator 
    'beta1': 0.5, # Momentum parameter
    'beta2': 0.999, # RMSprop parameter
    'save_epoch' : 20, # After how many epochs to save checkpoints and generate test output.
    'dataset' : 'ChestXRay', # Choose from {MNIST, SVHN, CelebA, FashionMNIST, ChestXRay}
    'print_model_description' : True, # Print the description of the models during training and generation
    'image_log': False # Save a generated image at every epoch (if False every 10 epochs)
}