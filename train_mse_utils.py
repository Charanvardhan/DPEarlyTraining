"""
Description: Training related functions

"""

#Some imports
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from flax import linen as nn
from flax import core
from flax import struct
from jax.numpy.linalg import norm

from data_utils_pytorch import estimate_num_batches

class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node = False)
    params: core.FrozenDict[str, Any]
    opt: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value."""
        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step = self.step + 1, params = new_params, opt_state = new_opt_state, **kwargs,)

    def update_learning_rate(self, *, learning_rate):
        """ Updates the learning rate"""
        self.opt_state.hyperparams['learning_rate'] = learning_rate
        return self.opt_state

    @classmethod
    def create(cls, *, apply_fn, params, opt, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = opt.init(params)
        return cls(step = 0, apply_fn = apply_fn, params = params, opt = opt, opt_state = opt_state, **kwargs, )

# train_mse_utils.py

import jax.numpy as jnp

def mse_loss(logits, y):
    # Ensure both logits and y are JAX arrays
    logits = jnp.asarray(logits)
    y = jnp.asarray(y)
    return 0.5 * jnp.mean((logits - y) ** 2)


def accuracy(logits, targets):
    """ Accuracy, used while measuring the state"""
    # Get the label of the one-hot encoded target
    target_class = jnp.argmax(targets, axis = 1)
    # Predict the class of the batch of images using
    predicted_class = jnp.argmax(logits, axis = 1)
    return jnp.mean(predicted_class == target_class)


@jax.jit
def train_batch(state, batch, gradient_computer, batch_size):
    "Compute gradients, loss and accuracy for a single batch"
    #print('Compiling train batch')
    x, y = batch

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x) 
        loss = mse_loss(logits, y)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)

     # Compute clipped and noisy gradients for DP
    clipped_noisy_grads, _ = gradient_computer.compute_gradients(grads, batch_size=batch_size)

    #update the state
    state = state.apply_gradients(grads = clipped_noisy_grads)
    return state, loss

@jax.jit
def hessian_batch(state, batch, power_iterations = 20):
    "Compute top eigenvalue and hessian"
    #print('Compiling Hessian batch')
    x, y = batch

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x) 
        loss = mse_loss(logits, y)
        return loss, logits

    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]

    body_hvp = jax.tree_util.Partial(hvp, flat_params)

    #  i here is only for fori_loop usage
    def fori_hvp(i, v):
        return body_hvp(v / norm(v))

    # Power Iteration
    key = jax.random.PRNGKey(24)
    v = jax.random.normal(key, shape=flat_params.shape)
    v = v / norm(v)
    v = jax.lax.fori_loop(0, power_iterations, fori_hvp, v / norm(v))
    top_eigen_value = jnp.vdot(v / norm(v), hvp(flat_params, v / norm(v)))

    return top_eigen_value


def measure_state(state, batches, num_train, batch_size):
    """
    Description: Estimates the loss and accuracy of a batched data stream

    Input:
    	1. state: a Trainstate instance
    	2. batches: a batched datastream
    	3. num_train: number of batches
    	4. batch_size: batch size

    """
    total_loss = 0
    total_accuracy = 0

    num_batches = estimate_num_batches(num_train, batch_size)

    for batch_ix in range(num_batches):
        batch = next(batches)
        x, y = batch
        #calculate logits
        logits = state.apply_fn({'params': state.params}, x)
        #calculate loss and accuracy
        total_loss += mse_loss(logits, y)
        total_accuracy += accuracy(logits, y)

    ds_loss = total_loss / num_batches
    ds_accuracy = total_accuracy / num_batches
    return ds_loss, ds_accuracy

def measure_fnorm(state, batches, num_train, batch_size):
    """
    Description: Estimates the loss and accuracy of a batched data stream

    Input:
        1. state: a Trainstate instance
        2. batches: a batched datastream
        3. num_train: number of batches
        4. batch_size: batch size

    """
    total_logits_norm = 0

    num_batches = estimate_num_batches(num_train, batch_size)

    for batch_ix in range(num_batches):
        batch = next(batches)
        x, y = batch
        #calculate logits
        logits = state.apply_fn({'params': state.params}, x)
        #calculate loss and accuracy
        print(logits.shape)
        total_logits_norm += jnp.linalg.norm(logits, axis = 1).mean()
        #total_logits_norm += mse_loss(logits, 0)**0.5
    ds_norm = total_logits_norm / num_batches
    return ds_norm

def measure_state_init(state, batches, num_train, batch_size):
    """
    Description: Estimates the loss and accuracy of a batched data stream

    Input:
        1. state: a Trainstate instance
        2. batches: a batched datastream
        3. num_train: number of batches
        4. batch_size: batch size

    """
    total_loss = 0
    total_accuracy = 0
    total_logits_norm = 0

    num_batches = estimate_num_batches(num_train, batch_size)

    for batch_ix in range(num_batches):
        batch = next(batches)
        x, y = batch
        #calculate logits
        logits = state.apply_fn({'params': state.params}, x)
        #calculate loss and accuracy
        total_logits_norm += mse_loss(logits, 0)
        total_loss += mse_loss(logits, y)
        total_accuracy += accuracy(logits, y)
    
    ds_norm = total_logits_norm / num_batches
    ds_loss = total_loss / num_batches
    ds_accuracy = total_accuracy / num_batches
    return ds_norm, ds_loss, ds_accuracy

def convert_to_jax(tensor):
    """Convert a PyTorch tensor to a JAX array."""
    if isinstance(tensor, jnp.ndarray):
        # If it's already a JAX array, return it directly
        return tensor
    else:
        # Convert PyTorch tensor to JAX array
        return jnp.array(tensor.detach().cpu().numpy())
    

def estimate_hessian(state, batches, num_batches = 10, power_iterations = 20):
    top_hessian = 0
    for i in range(num_batches):
        batch = next(batches)

        # Convert the batch to JAX tensors
        batch_images_jax = convert_to_jax(batch[0])
        batch_labels_jax = convert_to_jax(batch[1])

        # Now pass the JAX tensors to JAX operations
        top_hessian_batch = hessian_batch(state, (batch_images_jax, batch_labels_jax), power_iterations)
        top_hessian += top_hessian_batch
    top_hessian = top_hessian / num_batches
    return top_hessian
