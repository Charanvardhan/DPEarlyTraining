#!/usr/bin/env python
# coding: utf-8

# ### Installations

# Uncomment if needed for Colab
# !nvidia-smi -L
# !pip install -q jax flax optax ml_collections

import os
import data_utils_pytorch
import model_utils as models
import train_mse_utils as train_utils
from train_mse_utils import train_batch
import jax
from jax import numpy as jnp
import optax
import numpy as np
import pandas as pd

# Helper function to create and initialize the model
def create_train_state(config):
    model = models.Myrtle(num_filters=config['width'], num_layers=config['depth'], num_classes=config['num_classes'])
    example_input = jax.random.normal(config['init_rng'], shape=(1, 32, 32, 3))  # CIFAR-10 dimensions
    init_params = model.init(config['init_rng'], example_input)['params']
    optimizer = optax.sgd(learning_rate=config['lr'], momentum=config['momentum'])
    state = train_utils.TrainState.create(apply_fn=model.apply, params=init_params, opt=optimizer)
    return state

# Configuration setup
def create_config(width=128, depth=5, num_classes=10, lr=0.1, momentum=0.9, batch_size=64, num_epochs=1):
    return {
        'width': width,
        'depth': depth,
        'num_classes': num_classes,
        'lr': lr,
        'momentum': momentum,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'init_rng': jax.random.PRNGKey(0),
        'sgd_rng': jax.random.PRNGKey(0),
    }

# Training and evaluation function
def train_and_evaluate(config, train_ds, test_ds, iteration_count, save_dir="training_results"):
    """
    Description: Trains the model and evaluates on the test set.
    Inputs:
        - config: Dictionary of hyperparameters.
        - train_ds: Tuple (x_train, y_train).
        - test_ds: Tuple (x_test, y_test).
        - iteration_count: Number of training iterations.
        - save_dir: Directory to save CSV results.
    """
    train_results = []
    test_results = []
    
    rng = config['sgd_rng']
    train_batches = data_utils_pytorch.data_stream(rng, train_ds, config['batch_size'])
    test_batches = data_utils_pytorch.data_stream(rng, test_ds, config['batch_size'])
    state = create_train_state(config)
    
    train_loss_init, train_acc_init = train_utils.measure_state(state, train_batches, config['num_train'], config['measure_examples'])
    train_results.append([0, 0, train_loss_init, train_loss_init, train_acc_init])  # Initial values
    
    divergence = False

    # Training loop
    for epoch in range(config['num_epochs']):
        for step in range(iteration_count):
            batch = next(train_batches)
            step_num = epoch * iteration_count + step + 1

            # Training step
            state, loss_batch = train_utils.train_batch(state, batch)

            # Check for divergence
            if loss_batch > 10 ** 5:
                divergence = True
                print("Divergence detected, stopping training.")
                break

            # Measure and record training loss and accuracy
            train_loss_step, train_acc_step = train_utils.measure_state(state, train_batches, config['num_train'], config['measure_examples'])
            train_results.append([step_num, epoch + 1, train_loss_step, train_loss_init, train_acc_step])

        # Test evaluation after each epoch
        test_loss, test_acc = train_utils.measure_state(state, test_batches, config['num_test'], config['measure_examples'])
        test_results.append([epoch + 1, test_loss, test_acc])

        if divergence:
            break

    # Save training and testing results to CSV
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_filename = os.path.join(save_dir, f'train_results_{iteration_count}.csv')
    test_filename = os.path.join(save_dir, f'test_results_{iteration_count}.csv')

    pd.DataFrame(train_results, columns=['step', 'epoch', 'train_loss_step', 'train_loss_init', 'train_accuracy']).to_csv(train_filename, index=False)
    pd.DataFrame(test_results, columns=['epoch', 'test_loss', 'test_accuracy']).to_csv(test_filename, index=False)

    print(f"Training data saved to {train_filename}")
    print(f"Testing data saved to {test_filename}")

    return divergence, train_results, test_results

# ### Main Execution

# Hyperparameters and data loading
dataset = 'cifar10'
num_classes = 10
train_ds, test_ds, info = data_utils_pytorch.load_data_pytorch(dataset, num_classes)

config = create_config(width=128, depth=10, num_classes=num_classes, lr=0.1, momentum=0.9, batch_size=512, num_epochs=1)
config['num_train'] = info['num_train']
config['num_test'] = info['num_test']
config['measure_examples'] = 512

# Run training and save data
iteration_counts = [1000, 10000]  # Training for 1000, 10000, and 100000 iterations

for count in iteration_counts:
    print(f"\nTraining for {count} iterations...")
    train_and_evaluate(config, train_ds, test_ds, iteration_count=count)
