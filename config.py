# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'num_epochs': 10,# Number of epochs to train for. 
    'learning_rate': 2e-4,# Learning rate.
    'beta1': 0.5, # Momentum parameter
    'beta2': 0.999, #RMSprop parameter
    'save_epoch' : 10,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'MNIST'# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. Case sensitive!
}