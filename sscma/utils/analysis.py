# copyright Copyright (c) Seeed Technology Co.,Ltd.
from typing import Optional, Union

import torch
from mmengine.analysis.print_helper import (
    ActivationAnalyzer,
    FlopAnalyzer,
    _format_size,
    complexity_stats_str,
    complexity_stats_table,
    parameter_count,
)


def get_model_complexity_info(
    model: torch.nn.Module,
    input_shape: tuple,
    inputs: Optional[torch.Tensor] = None,
    show_table: bool = True,
    show_arch: bool = True,
    device: Optional[Union[torch.device, str]] = None,
):
    """Interface to get the complexity of a model.

    Args:
        model (nn.Module): The model to analyze.
        input_shape (tuple): The input shape of the model.
        inputs (torch.Tensor, optional): The input tensor of the model.
            If not given the input tensor will be generated automatically
            with the given input_shape.
        show_table (bool): Whether to show the complexity table.
            Defaults to True.
        show_arch (bool): Whether to show the complexity arch.
            Defaults to True.
        device (torch.device, str, optional): Setting the device to be
            used for inference

    Returns:
        dict: The complexity information of the model.
    """
    if inputs is None:
        inputs = (torch.randn(1, *input_shape, device=(device)),)
    else:
        inputs = inputs.to(device=device)

    flop_handler = FlopAnalyzer(model, inputs)
    activation_handler = ActivationAnalyzer(model, inputs)

    flops = flop_handler.total()
    activations = activation_handler.total()
    params = parameter_count(model)['']

    flops_str = _format_size(flops)
    activations_str = _format_size(activations)
    params_str = _format_size(params)

    if show_table:
        complexity_table = complexity_stats_table(
            flops=flop_handler,
            activations=activation_handler,
            show_param_shapes=True,
        )
        complexity_table = '\n' + complexity_table
    else:
        complexity_table = ''

    if show_arch:
        complexity_arch = complexity_stats_str(
            flops=flop_handler,
            activations=activation_handler,
        )
        complexity_arch = '\n' + complexity_arch
    else:
        complexity_arch = ''

    return {
        'flops': flops,
        'flops_str': flops_str,
        'activations': activations,
        'activations_str': activations_str,
        'params': params,
        'params_str': params_str,
        'out_table': complexity_table,
        'out_arch': complexity_arch,
    }
