#!/usr/bin/env python
# coding: utf-8

# ### Installations

# get_ipython().system('nvidia-smi -L')
# get_ipython().system('pip install -q jax flax optax ml_collections')


# ### Optional: Load Google drive and change directory to neurips_2023_demo

# import sys

# if "google.colab" in sys.modules:
#     print("Running on Google Colab")

#     from google.colab import drive
#     drive.mount('/content/drive')
#     %cd /content/drive/MyDrive/neurips_2023_15410/


# ### Load libraries

import os
import data_utils_pytorch
import model_utils as models
from train_mse_utils import train_batch
import jax
from jax import numpy as jnp
import optax
from ml_collections import config_dict

# usual imports
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# if loss == 'mse':
import train_mse_utils as train_utils
# ### Helper functions

def poly(coeffs, x):
    "Given polynomial coefficients coeffs, evaluates the polynomial f(x)"
    degree = len(coeffs) - 1
    output = 0
    for i in range(len(coeffs)):
        output += coeffs[i] * x ** (degree - i)
    return output


# pandas helper functions

def get_rows_where_col_equals(df, col, value):
    return df.loc[df[col] == value].copy()


def get_rows_where_col_in(df, col, values):
    return df.loc[df[col].isin(values)].copy()


# Helper functions for loss interpolation experiment

def interpolation_batch(init_params, final_params, state, batch):
    "Interpolates loss between initial and final parameters for a training batch"
    dt = 1e-02
    x, y = batch
    flat_init_params, rebuild_fn = jax.flatten_util.ravel_pytree(init_params)
    flat_final_params, rebuild_fn = jax.flatten_util.ravel_pytree(final_params)

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x)
        loss = train_utils.mse_loss(logits, y)
        return loss, logits

    def loss_fn_flat(flat_params, rebuild_fn):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    init_loss = loss_fn_flat(flat_init_params, rebuild_fn)
    final_loss = loss_fn_flat(flat_final_params, rebuild_fn)

    results = list()
    tau = jnp.arange(0, 1.0 + dt, dt)

    for t in tau:
        flat_params = t * flat_final_params + (1 - t) * flat_init_params
        loss = loss_fn_flat(flat_params, rebuild_fn)
        result = jnp.array([t, loss, init_loss, final_loss])
        results.append(result)

    return jnp.array(results)


def interpolate_params(init_params, final_params, state, batches, num_batches=10):
    dfs = list()
    for batch_ix in range(num_batches):
        batch = next(batches)
        result_batch = interpolation_batch(init_params, final_params, state, batch)
        dfs.append(result_batch)
    dfs = jnp.array(dfs)
    dfs = jnp.mean(dfs, axis=0)
    return dfs


# ### Model definition

def create_train_state(config, batches):
    """
    Description: Creates a Flax train state with learning rate eta = c / sharpness init
    Input: 
        1. config: ml_collections dict which contains the model and optimizer hyperparameters
        2. batches: batches used for sharpness estimation
    Output:
        1. state: Flax state with learning rate eta = c / sharpness init
        2. sharpness_init: sharpness at initialization
    """

    # Instantiate the Myrtle CNN model with the specified width and depth from model_utils
    model = models.Myrtle(num_filters=config.width, num_layers=config.depth, num_classes=config.num_classes)
    
    # Example input for initialization, use (32, 32, 3) for CIFAR-10 image dimensions
    example = jax.random.normal(config.init_rng, shape=(1, 32, 32, 3))

    # Initialize model parameters
    init_params = model.init(config.init_rng, example)['params']

    # Optimizer setup
    _opt = optax.sgd(learning_rate=0.1, momentum=config.momentum)
    _state = train_utils.TrainState.create(apply_fn=model.apply, params=init_params, opt=_opt)

    # Estimate sharpness at initialization using the helper function
    sharpness_init = train_utils.estimate_hessian(_state, batches, num_batches=config.measure_batches, 
                                                  power_iterations=config.power_iterations)

    # Set learning rate dynamically based on sharpness_init
    lr_rate = config.lr_const / sharpness_init
    opt = optax.sgd(learning_rate=lr_rate, momentum=config.momentum)
    state = train_utils.TrainState.create(apply_fn=model.apply, params=init_params, opt=opt)

    return state, sharpness_init

# ### Save & Load Training Data

def save_training_data(train_results, dfs, dfs_barrier, save_dir="training_data_10_width128_b512_h4098"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_results_df = pd.DataFrame(train_results, columns=['step', 'epoch', 'train_loss_step', 'train_loss_init',
                                                            'train_accuracy', 'sharpness_step', 'sharpness_init'])
    train_results_df.to_csv(os.path.join(save_dir, 'train_results.csv'), index=False)

    dfs.to_csv(os.path.join(save_dir, 'dfs.csv'), index=False)
    dfs_barrier.to_csv(os.path.join(save_dir, 'dfs_barrier.csv'), index=False)
    print(f"Data saved in '{save_dir}'")


def load_training_data(save_dir="training_data_10_width128_b512_h4098"):
    train_results_df = pd.read_csv(os.path.join(save_dir, 'train_results.csv'))
    dfs = pd.read_csv(os.path.join(save_dir, 'dfs.csv'))
    dfs_barrier = pd.read_csv(os.path.join(save_dir, 'dfs_barrier.csv'))
    return train_results_df, dfs, dfs_barrier


# ### Training and evaluation

def train_and_evaluate(config, train_ds):
    """
    Description: Creates a training state and trains for 10 steps
    Input: 
        1. config: ml_collections dictionary containing all the hyperparameters
        2. train_ds: tuple (x_train, y_train) of the training data
    Output: 
        1. Divergence: bool; Flag for training divergence
        2. train_results: np array containing training loss, sharpness and accuracy trajectories
    """

    train_results = list()

    rng = config.sgd_rng
    train_batches = data_utils_pytorch.data_stream(rng, train_ds, config.measure_examples)

    # Create training state with the Myrtle CNN model
    state, sharpness_init = create_train_state(config, train_batches)
    init_params = state.params

    # Measure initial loss and accuracy
    train_loss_init, train_acc_init = train_utils.measure_state(state, train_batches, config.num_train, config.measure_examples)

    result_init = jnp.asarray([0, 0, train_loss_init, train_loss_init, train_acc_init, sharpness_init, sharpness_init])
    train_results.append(result_init)

    divergence = False

    for epoch in range(config.num_epochs):

        rng, _ = jax.random.split(rng)
        batches = data_utils_pytorch.data_stream(rng, train_ds, config.batch_size)

        for batch_ix in range(config.num_steps):
            batch = next(batches)
            step = config.num_batches * epoch + batch_ix

            # Training step for the CNN model
            state, loss_batch = train_utils.train_batch(state, batch)

            # Check for divergence
            if loss_batch > 10 ** 5:
                divergence = True
                break

            # Measure training loss, accuracy, and sharpness
            train_loss_step, train_acc_step = train_utils.measure_state(state, train_batches, config.num_train, config.measure_examples)
            sharpness_step = train_utils.estimate_hessian(state, batches, num_batches=config.measure_batches, 
                                                          power_iterations=config.power_iterations)

            result_step = jnp.asarray([step + 1, epoch + 1, train_loss_step, train_loss_init, train_acc_step, sharpness_step, sharpness_init])
            train_results.append(result_step)

    int_results = list()

    if not divergence:
        final_params = state.params
        int_results = interpolate_params(init_params, final_params, state, train_batches, num_batches=config.measure_batches)

    del state
    train_results = jnp.asarray(train_results)
    train_results = jax.device_get(train_results)
    int_results = jax.device_get(int_results)

    return divergence, train_results, int_results

# ### Plot and save graphs
def plot_and_save_graphs(dfs, dfs_barrier, widths):
    lr_plot = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    dfs['lr_exp'] = dfs['lr_exp'].round(1)
    dfs['lr_const'] = dfs['lr_const'].round(1)

    df_plot = get_rows_where_col_in(dfs, 'lr_exp', lr_plot)
    print(df_plot['lr_exp'].unique())

    for i, width in enumerate(widths):
        df_width = get_rows_where_col_equals(df_plot, 'width', width)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax = sns.lineplot(x='step', y='train_loss_step', data=df_width, hue='lr_const', palette='crest', legend='full', ax=ax)
        ax.set_xlabel('step')
        ax.set_ylabel(r'Training loss')
        ax.set_title(f'width: {width}')

        # Only remove the legend if it exists
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax = axes[1]
        ax = sns.lineplot(x='step', y='norm_sharp', data=df_width, hue='lr_const', palette='crest', legend='full', ax=ax)
        ax.set_xlabel('step')
        ax.set_ylabel(r'$\frac{\lambda_t^H}{\lambda_0^H}$')
        ax.set_title(f'width: {width}')

        # Ensure legend exists before trying to display it
        if ax.get_legend() is not None:
            ax.legend(title=r'$c$', loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(f'training_trajectories_width_{width}.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_and_save_graphs_10000(dfs, dfs_barrier, widths):
    lr_plot = [1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0, 11.3, 16.0, 22.6]  # As in the screenshot

    dfs['lr_exp'] = dfs['lr_exp'].round(1)
    dfs['lr_const'] = dfs['lr_const'].round(1)
    # Add sharpness normalization if not present
    if 'norm_sharp' not in dfs.columns:
        dfs['norm_sharp'] = dfs['sharpness_step'] / dfs['sharpness_init']
    df_plot = dfs[dfs['lr_exp'].isin(lr_plot)]
    df_plot['step'] = np.log10(df_plot['step'])

    for width in widths:
        df_width = df_plot[df_plot['width'] == width]

        # Create subplots for (a) training loss, (b) sharpness, and (c) training accuracy
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) Training Loss
        ax = axes[0]
        sns.lineplot(x='step', y='train_loss_step', data=df_width, hue='lr_const', palette='crest', ax=ax)
        ax.set_xlabel('log10(t) step')
        ax.set_ylabel('Training loss')
        ax.set_title('(a) Training loss')
        ax.legend(title=r'$c$', loc='upper right')

        # (b) Sharpness λt^H / λ0^H
        ax = axes[1]
        sns.lineplot(x='step', y='norm_sharp', data=df_width, hue='lr_const', palette='crest', ax=ax)
        ax.set_xlabel('log10(t) step')
        ax.set_ylabel(r'$\lambda_t^H / \lambda_0^H$')
        ax.set_title('(b) Sharpness')
        ax.axhline(2.0, linestyle='--', color='gray')  # Threshold
        ax.legend(title=r'$c$', loc='upper right')

        # (c) Training Accuracy
        ax = axes[2]
        sns.lineplot(x='step', y='train_accuracy', data=df_width, hue='lr_const', palette='crest', ax=ax)
        ax.set_xlabel('log10(t) step')
        ax.set_ylabel('Training accuracy')
        ax.set_title('(c) Training accuracy')
        ax.legend(title=r'$c$', loc='upper right')

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'training_trajectories_width_{width}.png', dpi=300, bbox_inches='tight')
        plt.show()



# ### Hyperparameters

dataset = 'cifar10'
num_classes = 10

train_ds, test_ds, info = data_utils_pytorch.load_data_pytorch(dataset, num_classes)

# Hyperparameters
config = config_dict.ConfigDict()
config.num_train, config.num_test = info['num_train'], info['num_test']

config.act = 'relu'  # Activation function
config.use_bias = False
config.varw = 2.0  # Variance scaling for initialization
config.varwL = 1.0
widths = [128]
# CNN-specific settings
config.depth = 5  # Myrtle CNN depth
# config.width = 512  # Myrtle CNN width (num_filters)
config.in_dim = info['in_dim']
config.num_classes = info['num_classes'] # CIFAR-10 has 10 classes

config.batch_size = 512
config.measure_examples = 4096  # Examples for sharpness measurement
config.measure_batches = 1
config.num_batches = data_utils_pytorch.estimate_num_batches(config.num_train, config.batch_size)
config.power_iterations = 20  # For sharpness estimation

lr_exp_start = jax.device_put(1.0)
lr_step = 0.4
config.momentum = jax.device_put(0.0)
config.num_steps = 10
config.num_epochs = 5  # Number of epochs


init_averages = 1
sgd_runs = 1

# ### Run experiment and save data

dfs = list()
dfs_barrier = list()

for width in widths:
    config.width = width
    for iteration in range(1, init_averages + 1):
        config.init_rng = jax.random.PRNGKey(iteration)
        for run in range(1, sgd_runs + 1):
            config.sgd_rng = jax.random.PRNGKey(run)
            lr_exp = lr_exp_start
            divergence = False

            while not divergence:
                config.lr_exp = lr_exp
                config.lr_const = 2 ** lr_exp
                print(f'w: {config.width}, d: {config.depth}, I: {iteration}, J: {run}, x: {lr_exp:0.1f}, B: {config.batch_size}, t: {config.num_steps}')

                divergence, train_results, int_results = train_and_evaluate(config, train_ds)

                if not divergence:
                    df = pd.DataFrame(train_results, columns=['step', 'epoch', 'train_loss_step', 'train_loss_init', 'train_accuracy', 'sharpness_step', 'sharpness_init'], dtype=float)
                    df['lr_exp'] = config.lr_exp
                    df['lr_const'] = config.lr_const
                    df['batch_size'] = config.batch_size
                    df['num_steps'] = config.num_epochs
                    df['I'] = iteration
                    df['J'] = run
                    df['width'] = config.width
                    df['depth'] = config.depth
                    dfs.append(df)

                    df = pd.DataFrame(int_results, columns=['tau', 'int_loss', 'init_loss', 'final_loss'])
                    df['lr_exp'] = config.lr_exp
                    df['lr_const'] = config.lr_const
                    df['batch_size'] = config.batch_size
                    df['num_steps'] = config.num_steps
                    df['I'] = iteration
                    df['J'] = run
                    df['width'] = config.width
                    df['depth'] = config.depth
                    dfs_barrier.append(df)

                else:
                    print('Divergence')

                lr_exp += lr_step
                gc.collect()

dfs = pd.concat(dfs, axis=0, ignore_index=True)
dfs_barrier = pd.concat(dfs_barrier, axis=0, ignore_index=True)

save_training_data(train_results, dfs, dfs_barrier)
# plot_and_save_graphs(dfs, dfs_barrier, widths)
# Call the function
plot_and_save_graphs_10000(dfs, dfs_barrier, widths)

