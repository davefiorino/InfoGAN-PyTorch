# Dictionary storing network parameters.
params = {
    'batch_size': 256, # Batch size.
    'num_epochs': 100, # Number of epochs to train for. 
    'learning_rate': 2e-4, # Learning rate.
    'beta1': 0.5, # Momentum parameter
    'beta2': 0.999, # RMSprop parameter
    'save_epoch' : 20, # After how many epochs to save checkpoints and generate test output.
    'dataset' : 'ChestXRay', # Choose from {MNIST, SVHN, CelebA, FashionMNIST, ChestXRay}
    'print_model_description' : True, # Print the description of the models during training and generation
    'image_log': False # Save a generated image at every epoch (if False every 10 epochs)
}