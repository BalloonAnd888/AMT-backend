import torch 

def init_weights(module, init_fn=torch.nn.init.kaiming_normal,
                 bias_val=0.0, verbose=False):
    """
    Custom, layer-aware initializer for PyTorch modules.

    :param init_fn: initialization function, such that ``init_fn(weight)``
      modifies in-place the weight values. If ``None``, found weights won't be
      altered
    :param float bias_val: Any module with biases will initialize them to this
      constant value

    Usage example, inside of any ``torch.nn.Module.__init__`` method:

    if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))

    Apply is applied recursively to any submodule inside, so this works.
    """
    if isinstance(module, (torch.nn.Linear,
                           torch.nn.Conv1d,
                           torch.nn.Conv2d)):
        if init_fn is not None:
            init_fn(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(bias_val)
    elif isinstance(module, (torch.nn.GRU, torch.nn.LSTM)):
        raise NotImplementedError("No RNNs supported at the moment :)")
    else:
        if verbose:
            print("init_weights: ignored module:", module.__class__.__name__)