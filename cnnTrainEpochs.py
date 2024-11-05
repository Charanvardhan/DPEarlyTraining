import os
import pandas as pd
import data_utils_pytorch
import model_utils as models
import train_mse_utils as train_utils
import jax
from jax import numpy as jnp
import optax

# Helper function to create and initialize the model
def create_train_state(config):
    model = models.Myrtle(num_filters=config['width'], num_layers=config['depth'], num_classes=config['num_classes'])
    example_input = jax.random.normal(config['init_rng'], shape=(1, 32, 32, 3))  # CIFAR-10 dimensions
    init_params = model.init(config['init_rng'], example_input)['params']
    optimizer = optax.sgd(learning_rate=config['lr'], momentum=config['momentum'])
    state = train_utils.TrainState.create(apply_fn=model.apply, params=init_params, opt=optimizer)
    return state

# Configuration setup
def create_config(width=512, depth=5, num_classes=10, lr=0.1, momentum=0.9, batch_size=64, num_epochs=1):
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

# Training and evaluation function based on epochs and iterations
def train_and_evaluate(config, train_ds, test_ds, output_file="all_train_results.csv"):
    """
    Description: Trains the model, evaluates on the test set, and appends results to a single CSV file.
    Inputs:
        - config: Dictionary of hyperparameters.
        - train_ds: Tuple (x_train, y_train).
        - test_ds: Tuple (x_test, y_test).
        - output_file: Path to the cumulative CSV file.
    """
    train_results = []
    
    rng = config['sgd_rng']
    train_batches = data_utils_pytorch.data_stream(rng, train_ds, config['batch_size'])
    test_batches = data_utils_pytorch.data_stream(rng, test_ds, config['batch_size'])
    state = create_train_state(config)
    
    # Calculate the number of iterations per epoch based on batch size and dataset size
    num_train_batches = config['num_train'] // config['batch_size']
    if config['num_train'] % config['batch_size'] != 0:
        num_train_batches += 1  # Add an extra batch if there's a remainder
    
    train_loss_init, train_acc_init = train_utils.measure_state(state, train_batches, config['num_train'], config['measure_examples'])
    
    divergence = False

    # Training loop
    for epoch in range(config['num_epochs']):
        for step in range(num_train_batches):
            batch = next(train_batches)
            step_num = epoch * num_train_batches + step + 1

            # Training step
            state, loss_batch = train_utils.train_batch(state, batch)

            # Check for divergence
            if loss_batch > 10 ** 5:
                divergence = True
                print("Divergence detected, stopping training.")
                break

        # Measure and record training loss and accuracy at the end of each epoch
        train_loss_epoch, train_acc_epoch = train_utils.measure_state(state, train_batches, config['num_train'], config['measure_examples'])
        train_results.append([config['width'], config['depth'], epoch + 1, train_loss_epoch, train_loss_init, train_acc_epoch, 'training'])

        if divergence:
            break

    # Test evaluation after training is complete
    test_loss, test_acc = train_utils.measure_state(state, test_batches, config['num_test'], config['measure_examples'])
    train_results.append([config['width'], config['depth'], config['num_epochs'], test_loss, train_loss_init, test_acc, 'testing'])

    # Append training and test results to the cumulative CSV file
    train_df = pd.DataFrame(train_results, columns=['width', 'depth', 'epoch', 'loss', 'train_loss_init', 'accuracy', 'phase'])
    if not os.path.isfile(output_file):
        train_df.to_csv(output_file, index=False)
    else:
        train_df.to_csv(output_file, mode='a', header=False, index=False)
    
    print(f"Training and testing data for width={config['width']}, depth={config['depth']} saved to {output_file}")

    return train_results

# ### Main Execution

# Hyperparameters and data loading
dataset = 'cifar10'
num_classes = 10
train_ds, test_ds, info = data_utils_pytorch.load_data_pytorch(dataset, num_classes)

widths = [128, 256, 512, 1024]
depths = [5, 10]
output_file = "all_train_results_0.005.csv"

# Loop through model configurations and train
for width in widths:
    for depth in depths:
        config = create_config(width=width, depth=depth, num_classes=num_classes, lr=0.005, momentum=0.9, batch_size=512, num_epochs=50)
        config['num_train'] = info['num_train']
        config['num_test'] = info['num_test']
        config['measure_examples'] = 512

        # Run training and append data to the cumulative CSV file
        print(f"\nTraining with width={width}, depth={depth} for {config['num_epochs']} epochs...")
        train_and_evaluate(config, train_ds, test_ds, output_file=output_file)
