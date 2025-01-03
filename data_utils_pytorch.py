import numpy as np
import torch
import jax.numpy as jnp
import jax
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ml_collections import config_dict

def _one_hot_pytorch(x, k, dtype=torch.float64):
    """Create a one-hot encoding of x of size k."""
    x = x.long()  # Convert to long tensor to be compatible with one-hot encoding
    return F.one_hot(x, num_classes=k).type(dtype)

def _standardize_pytorch(x):
    """Standardization per sample across feature dimension."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std_dev = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - mean) / std_dev

# def load_data_pytorch(dataset, num_classes):
#     """
#     Loads and preprocesses common image datasets using PyTorch's torchvision.
#     Supports: CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST.
#     """
#     # Define transforms: normalization and standardization
#     if dataset == 'cifar10':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for RGB channels
#         ])

#     # Load datasets using torchvision
#     if dataset == 'cifar10':
#         train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     elif dataset == 'cifar100':
#         train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
#     elif dataset == 'mnist':
#         train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     elif dataset == 'fashion_mnist':
#         train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
#     else:
#         raise ValueError(f"Unsupported dataset {dataset}")

#     # Use DataLoader to create batches
#     batch_size = 256
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#     # Process a single batch for shape and preprocessing checks
#     for x_train, y_train in train_loader:
#         x_train = x_train.to(torch.float64)
#         y_train = y_train.to(torch.float64)
#         break  # Only get the first batch to standardize and process

#     # Standardize input
#     x_train = _standardize_pytorch(x_train)

#     # Get one-hot encoding for the labels
#     y_train = _one_hot_pytorch(y_train, num_classes)

#     # Process the entire dataset
#     x_train = torch.cat([_standardize_pytorch(x_batch) for x_batch, _ in train_loader])
#     y_train = torch.cat([_one_hot_pytorch(y_batch, num_classes) for _, y_batch in train_loader])
#     x_test = torch.cat([_standardize_pytorch(x_batch) for x_batch, _ in test_loader])
#     y_test = torch.cat([_one_hot_pytorch(y_batch, num_classes) for _, y_batch in test_loader])

#     # Dataset information
#     info = config_dict.ConfigDict()
#     info.num_train = x_train.shape[0]
#     info.num_test = x_test.shape[0]
#     info.num_classes = num_classes
#     info.in_dim = (32, 32, 3)
#     print(info.in_dim, '================================')

#     return (x_train, y_train), (x_test, y_test), info


def load_data_pytorch(dataset, num_classes):
    """
    Loads and preprocesses common image datasets using PyTorch's torchvision.
    Converts the datasets to JAX-compatible arrays.
    """
    # Define transforms: normalization and standardization
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for RGB channels
        ])

    # Load datasets using torchvision
    if dataset == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar100':
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'mnist':
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'fashion_mnist':
        train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    # Use DataLoader to load data into batches
    batch_size = 256
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Convert all data to JAX-compatible arrays
    train_images = []
    train_labels = []
    for x_batch, y_batch in train_loader:
        train_images.append(jnp.array(x_batch.numpy()))
        train_labels.append(jnp.array(y_batch.numpy()))
    test_images = []
    test_labels = []
    for x_batch, y_batch in test_loader:
        test_images.append(jnp.array(x_batch.numpy()))
        test_labels.append(jnp.array(y_batch.numpy()))

    # Concatenate all batches
    train_images = jnp.concatenate(train_images)
    train_labels = jnp.concatenate(train_labels)
    test_images = jnp.concatenate(test_images)
    test_labels = jnp.concatenate(test_labels)

    # Dataset information
    info = config_dict.ConfigDict()
    info.num_train = train_images.shape[0]
    info.num_test = test_images.shape[0]
    info.num_classes = num_classes
    info.in_dim = (32, 32, 3)

    return (train_images, train_labels), (test_images, test_labels), info


def data_stream(seed, ds, batch_size):
    """Creates a data stream with a predefined batch size."""
    train_images, train_labels = ds
    num_train = len(train_images)
    num_batches = estimate_num_batches(num_train, batch_size)

    # Convert JAX PRNG key to integer seed for PyTorch RNG
    seed_int = int(jax.random.randint(seed, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed_int)  # Use NumPy RNG

    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]



def estimate_num_batches(num_train, batch_size):
    """Estimates number of batches using dataset and batch size."""
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches
