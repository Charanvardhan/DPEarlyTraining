"""Data loading utilities"""

# Some imports
import numpy as np
import jax.numpy as jnp
import jax
import numpy as np
from ml_collections import config_dict
import pickle as pl
import tensorflow_datasets as tfds

def _one_hot(x, k, dtype = jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def _standardize(x):
  """Standardization per sample across feature dimension."""
  axes = tuple(range(1, len(x.shape)))
  mean = jnp.mean(x, axis = axes, keepdims = True)
  std_dev = jnp.std(x, axis = axes, keepdims = True)
  return (x - mean) / std_dev

def load_data_pytorch(dataset, num_classes):
    """
    Loads common image datasets using PyTorch's torchvision.
    Supports: CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST
    """

    # Define transforms: normalization and standardization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Standardize across features
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

    # Use DataLoader to create batches
    batch_size = 512
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Get dataset info
    info = {
        'num_train': len(train_set),
        'num_test': len(test_set),
        'num_classes': num_classes,
        'in_dim': train_set[0][0].shape  # Input dimension from the first sample
    }

    return train_loader, test_loader, info


def load_data(dataset, num_classes):

    #path = f'datasets/{dataset}.dump'    
    #in_file = open(path, 'rb')

    if dataset in ['cifar100', 'cifar10', 'mnist', 'fashion_mnist']:
        ds_train, ds_test = tfds.as_numpy(tfds.load(dataset, data_dir = './',split = ["train", "test"], batch_size = -1, as_dataset_kwargs = {"shuffle_files": False}))
    else:
        raise ValueError("Invalid dataset name.")

    x_train, y_train, x_test, y_test = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])


    x_train = np.asarray(x_train, dtype=np.float32)  # Convert to NumPy array first
    x_train = jax.device_put(x_train)  # Then move to the GPU if needed

    y_train = np.asarray(x_train, dtype=np.float32)
    y_train = jax.device_put(y_train)

    x_test = np.asarray(x_train, dtype=np.float32)
    x_test = jax.device_put(x_test)
    y_test = np.asarray(y_test, dtype=np.float32)
    y_test = jax.device_put(y_test)

    #get info
    info = config_dict.ConfigDict()
    info.num_train = x_train.shape[0]
    info.num_test = x_test.shape[0]
    info.num_classes = num_classes
    info.in_dim = (1, *x_train[0].shape)

    #standardize input
    x_train, x_test = _standardize(x_train), _standardize(x_test)

    #get one hot encoding for the labels
    y_train = _one_hot(y_train, num_classes)
    y_test = _one_hot(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), info

def load_data_regress(dataset):
    """
    loads mnist dataset from tensorflow_datasets
    """

    path = f'datasets/{dataset}.dump'
    in_file = open(path, 'rb')
    (x_train, y_train), (x_test, y_test) = pl.load(in_file)

    #get info
    info = config_dict.ConfigDict()
    info.num_train = x_train.shape[0]
    info.num_test = x_test.shape[0]
    
    info.in_dim = x_train.shape[1:]
    info.out_dim = y_Train.shape[1:]

    return (x_train, y_train), (x_test, y_test), info

def data_stream(key, ds, batch_size):
    " Creates a data stream with a predifined batch size."
    train_images, train_labels = ds
    num_train = train_images.shape[0]
    num_batches = estimate_num_batches(num_train, batch_size)
    rng = np.random.RandomState(key)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1)*batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]

def estimate_num_batches(num_train, batch_size):
    "Estimates number of batches using dataset and batch size"
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches

