"""
Description: Training related functions

"""

#Some imports
import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from flax import linen as nn
from flax import core
from flax import struct
from jax.numpy.linalg import norm

from data_utils import estimate_num_batches

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

def xent_loss(logits, labels):
  return optax.softmax_cross_entropy(logits = logits, labels = labels).mean()

@jax.jit
def accuracy(logits, targets):
    """ Accuracy, used while measuring the state"""
    # Get the label of the one-hot encoded target
    target_class = jnp.argmax(targets, axis = 1)
    # Predict the class of the batch of images using
    predicted_class = jnp.argmax(logits, axis = 1)
    return jnp.mean(predicted_class == target_class)


@jax.jit
def train_batch(state, batch):
    "Compute gradients, loss and accuracy for a single batch"
    x, y = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x) 
        loss = xent_loss(logits, y)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    #update the state
    state = state.apply_gradients(grads = grads)
    return state, loss

@jax.jit
def hessian_batch(state, batch, power_iterations = 20):
    "Compute top eigenvalue and hessian"
    x, y = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x) 
        loss = xent_loss(logits, y)
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
        total_loss += xent_loss(logits, y)
        total_accuracy += accuracy(logits, y)

    ds_loss = total_loss / num_batches
    ds_accuracy = total_accuracy / num_batches
    return ds_loss, ds_accuracy

def estimate_hessian(state, batches, num_batches = 10, power_iterations = 20):
    top_hessian = 0
    for batch_ix in range(num_batches):
        batch = next(batches)
        x, y = batch
        top_hessian_batch = hessian_batch(state, batch, power_iterations)
        top_hessian += top_hessian_batch
    top_hessian = top_hessian / num_batches
    return top_hessian

