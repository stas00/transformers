# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import ceil

import torch


def validate_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    assert len(duplicate_blocks) == 0, (
        "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These "
        "attention blocks were specified more than once: " + str(duplicate_blocks)
    )
    assert len(missing_blocks) == 0, (
        "There are attention blocks for this model that are not specified in the device_map. Add these attention "
        "blocks to a device on the device_map: " + str(missing_blocks)
    )
    assert (
        len(extra_blocks) == 0
    ), "The device_map contains more attention blocks than this model has. Remove these from the device_map:" + str(
        extra_blocks
    )


def make_default_device_map(n_layers):
    """Returns a dictionary of layers distributed evenly across all devices."""
    n_gpus = torch.cuda.device_count()
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / n_gpus))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(range(n_gpus), layers_list))


def init_device_map(n_layers, device_map=None):
    """
    - creates a device_map if none was passed
    - validates that map is correct

    Args:
      n_layers - how many total layers to remap
    """
    if device_map is None:
        device_map = make_default_device_map(n_layers)
    validate_device_map(device_map, n_layers)
    return device_map

# get variable name (doesn't work for everything)
import inspect
def log_name_device(var, fallbackname=""): # search from the outmost frame inwards
    """
    This helper is useful for debug tracing of devices of variables, e.g.:
      logger.info(f"MP {self.__class__.__name__} {log_name_device(attention_mask)}")
    if it can't deduce the variable name (or finds wrong name), pass the name explicitly, e.g.:
      logger.info(f"MP {self.__class__.__name__} {log_name_device(self.lm_head, 'self.lm_head')}")
    """

    for f in reversed(inspect.stack()):
        name = "unknown"
        names = [x for x, val in f.frame.f_locals.items() if val is var]
        if len(names) > 0:
            name = names[0]
            break
    if name == "var":
        name = fallbackname

    device = None
    try:
        device = var.device
    except:
        try:
            device = get_layer_device(var)
        except:
            pass
    return f"{name} {device}"


def get_layer_device(self):
        try:
            device = next(self.parameters(recurse=True)).device
        except StopIteration:
            device = None
        return device

# def to_dev(self, input):
#         try:
#             device = next(self.parameters(recurse=True)).device
#         except StopIteration:
#             device = None

#         if device is None:
#             raise ValueError(f"Can't find any params for {self.__class__}")
#         print(f"manual switch to {device}")
#         return input.to(device)


def model_parallel_inputs_to_device(func):
    """
    This decorator will try to find a at least one parameter or a buffer to read layer's .device from and then will
    automatically copy any inputs to that device before `forward` is called.

    this will work do its magical thing only if all params of this layer are on the same device
    """

    def _call__mp(self, *input, **kwargs):

        if not hasattr(self, "model_parallel") or not self.model_parallel:
            return func(self, *input, **kwargs)

        # get device of any of the param of this layer
        try:
            device = next(self.parameters(recurse=True)).device
        except StopIteration:
            device = None

        # print(f"layer device: {device}")
        if device is not None:
            #torch.cuda.set_device(device)
            print(f"auto-move inputs to {device}")

            input = recursive_to(device, input)

            return func(self, *input, **kwargs)

    return _call__mp


def recursive_to(device, item):
    """
    Switch any tensors found in `item` to `device`.
    Currently can handle a single tensor, or any of the nested list, tuple and dict structures.
    """

    if torch.is_tensor(item):
        return item.to(device)

    elif isinstance(item, list):
        for i, x in enumerate(item):
            item[i] = recursive_to(device, x)
        return item

    elif isinstance(item, tuple):
        return tuple(recursive_to(device, list(item)))

    elif isinstance(item, dict):
        for k, v in item.items():
            item[k] = recursive_to(device, v)
        return item

    else:
        return item


def model_parallel_inputs_to_specific_device(device, *input):
    print(f"move specific inputs to {device}")
    output = recursive_to(device, input)
    # remove the need for the caller to perform "a, = foo(a)", which otherwise will make `a` a tuple when it might be not be
    return output[0] if len(output)==1 else output


def model_parallel_call(self, *input, **kwargs):
    
    # get device of any of the param of this layer
    try:
        device = next(self.parameters(recurse=True)).device
    except StopIteration:
        device = None

    # print(f"layer device: {device}")
    if device is not None:
        input = recursive_to(device, input)
        kwargs = recursive_to(device, kwargs)

    return nn.Module.__call__(self, *input, **kwargs)


# XXX: still used by gpt2 so leave here for now
assert_device_map = validate_device_map


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))
