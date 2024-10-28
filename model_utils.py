"""
Model definitions
"""
from flax import linen as nn
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import *
from functools import *



activations = {'relu': nn.relu, 'tanh': jnp.tanh}

#Usage model = fcn(width, depth, num_classes, use_bias, varw, act)

############################
####### Vanilla FCN ########
############################


class fcn(nn.Module):
    """
    Description: A constant width fully connected network in standard parameterization

    Inputs: 
        1. width: width of the network
        2. depth: depth of the network
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name

    """
    width: int
    depth: int
    out_dim: int
    use_bias: bool
    varw: float
    act_name: str

    def setup(self):
        # setup initialization for all but last layer
        kernel_init =  nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')
        # setup activation
        self.act = activations[self.act_name]
        # create a list of all but last layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer =  [nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = nn.initializers.variance_scaling(scale = 1.0, distribution = 'normal', mode = 'fan_in') )]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 != self.depth:
                x = self.act(x)
        return x        


class fcn_si(nn.Module):
    """
    Description: A constant width fully connected network in standard parameterization

    Inputs: 
        1. width: width of the network
        2. depth: depth of the network
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name

    """
    width: int
    depth: int
    out_dim: int
    use_bias: bool
    varw: float
    act_name: str

    def setup(self):
        # setup initialization for all but last layer
        kernel_init =  nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')
        # setup activation
        self.act = activations[self.act_name]
        # create a list of all but last layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer =  [nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = nn.initializers.variance_scaling(scale = 1.0 / self.width, distribution = 'normal', mode = 'fan_in') )]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 != self.depth:
                x = self.act(x)
        return x        


class fcn_zeros(nn.Module):
    """
    Description: A constant width fully connected network in standard parameterization

    Inputs: 
        1. width: width of the network
        2. depth: depth of the network
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name

    """
    width: int
    depth: int
    out_dim: int
    use_bias: bool
    varw: float
    act_name: str

    def setup(self):
        # setup initialization for all but last layer
        kernel_init =  nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')
        # setup activation
        self.act = activations[self.act_name]
        # create a list of all but last layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer =  [nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = nn.initializers.zeros )]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 != self.depth:
                x = self.act(x)
        return x        
class fcn_scale(nn.Module):
    """
    Description: A constant width fully connected network in standard parameterization

    Inputs:
        1. width: width of the network
        2. depth: depth of the network
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name

    """
    width: int
    depth: int
    out_dim: int
    use_bias: bool
    varw: float
    alpha: float
    act_name: str

    def setup(self):
        # setup initialization for all but last layer
        kernel_init =  nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')
        # setup activation
        self.act = activations[self.act_name]
        # create a list of all but last layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer =  [nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = nn.initializers.variance_scaling(scale = 1.0, distribution = 'normal', mode = 'fan_in') )]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1))
        #depth = len(self.layers)
        #print(depth, self.depth)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 != self.depth:
                x = self.act(x)
        x = x * self.alpha 
        return x


class fcn_mup(nn.Module):
    """
    Description: A constant width fully connected network in standard parameterization

    Inputs: 
        1. width: width of the network
        2. depth: depth of the network
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name

    """
    width: int
    depth: int
    out_dim: int
    use_bias: bool
    varw: float
    out_exp: float
    act_name: str

    def setup(self):
        # setup initialization for all but last layer
        kernel_init =  nn.initializers.variance_scaling(scale = self.varw, distribution = 'normal', mode = 'fan_in')
        # setup activation
        self.act = activations[self.act_name]
        # create a list of all but last layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer =  [nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = nn.initializers.variance_scaling(scale = 1.0, distribution = 'normal', mode = 'fan_in') )]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1))
        #depth = len(self.layers)
        #print(depth, self.depth)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 != self.depth:
                x = self.act(x)
        x = x / (self.width)**self.out_exp
        return x        

class NTPDense(nn.Module):
    "NTP Dense without bias"
    fan_out: int
    varw: float = 1.0
    
    @nn.compact
    def __call__(self, inputs):
        kernel_init = nn.initializers.normal(stddev = 1.0)
        fan_in = inputs.shape[-1]
        kernel = self.param('kernel', kernel_init, (fan_in, self.fan_out))  
        norm = jnp.sqrt(self.varw / fan_in)
        h = norm*jnp.dot(inputs, kernel) 
        return h

class linear_fcn(nn.Module):
    """
    Description: A constant width fully connected network in standard parameterization

    Inputs:
        1. width: width of the network
        2. depth: depth of the network
        3. out_dim: output dimension / number of classes
        4. use_bias: wheather to use bias or not
        5. varw: variance of the weights except the last layer; last layer always have varw = 1.0
        6. act_name: activation name

    """
    width: int
    depth: int
    out_dim: int
    alpha: float = 0.0 # the parameter s

    def setup(self):
        # setup initialization for all but last layer
        # create a list of all but last layer
        self.layers = [NTPDense(self.width) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer =  [NTPDense(self.out_dim)]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1))
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x / (self.width)**self.alpha
        return x



###########################################
############## Myrtle CNNs ################
###########################################

#dictionary of pool list and depth
model_pool_list = {
    5: [1, 2, 3],
    7: [1, 3, 5],
    10: [2, 5, 8]
    }
#model = Myrtle(config.width, config.depth, config.pool_list, config.num_classes)
#config.pool_list, config.depth = models[config.model]

ModuleDef = Any

class Block(nn.Module):
  """CNN block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  kernel: Tuple[int, int] = (3, 3)

  @nn.compact
  def __call__(self, x,):
    x = self.conv(self.filters, self.kernel)(x)
    x = self.act(x)
    return x

class Myrtle(nn.Module):
    """Myrtle CNNs"""
    num_filters: int
    num_layers: int
    num_classes: int = 10
    block_cls: ModuleDef = Block
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    use_bias: bool = False
    kernel: Tuple[int, int] = (3, 3)
    varw: float = 2.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # print(f"Input shape in Block (after transpose): {x.shape}")
        pool_list = model_pool_list[self.num_layers]
        varw = self.varw 
        kernel_init =  nn.initializers.variance_scaling(scale = varw, distribution = 'normal', mode = 'fan_in')
        conv = partial(self.conv, use_bias = False, dtype = self.dtype, padding = 'SAME', strides = 1, kernel_init = kernel_init)
        # normalization is not used in the Block; this is just for generalization of the implementation later
        norm = partial(nn.LayerNorm, use_bias = False, use_scale = False, dtype = self.dtype)
        for i in range(self.num_layers - 1):
            x = self.block_cls(self.num_filters, conv = conv, norm = norm, act = self.act, kernel = self.kernel)(x)
            if i in pool_list:
                x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        x = x.reshape((x.shape[0], -1))

        #default initialization in flax is lecun normal https://flax.readthedocs.io/en/latest/_modules/flax/linen/linear.html#Dense
        x = nn.Dense(self.num_classes, use_bias = False, dtype = self.dtype)(x)

        x = jnp.asarray(x, self.dtype)
        return x

######################################
############  ResNets ################
######################################

ModuleDef = Any
class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm(y)

    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    # y = self.norm(scale_init=nn.initializers.zeros)(y)
    y = self.norm(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      # residual = self.norm(name='norm_proj')(residual)
      residual = self.norm(residual)

    return self.act(residual + y)



class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    # y = self.norm()(y)
    #y = self.norm(y)

    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    # y = self.norm()(y)
    #y = self.norm(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    #y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      #residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)

class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = lambda x: x

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    # x = norm(name='bn_init')(x)
    x = norm(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock)
