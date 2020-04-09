from collections import OrderedDict
import networkx
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init


# TODO: Fix FeedForward Model
# class FeedforwardModel(nn.Module):
#
#     def __init__(self,
#                  model_str,
#                  model_kwargs):
#
#         super(FeedforwardModel, self).__init__()
#         self.input_size = model_kwargs['input_size']
#         self.output_size = model_kwargs['output_size']
#         self.model_str = model_str
#         self.model_kwargs = model_kwargs
#
#         # create and save feedforward network
#         self.feedforward = self._create_feedforward(
#             model_kwargs=model_kwargs)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.description_str = create_description_str(model=self)
#
#         # converts all weights into doubles i.e. float64
#         # this prevents PyTorch from breaking when multiplying float32 * float64
#         self.double()
#
#         # dummy_input = torch.zeros(size=(10, 1, 1), dtype=torch.double)
#         # tensorboard_writer.add_graph(
#         #     model=self,
#         #     input_to_model=dict(stimulus=dummy_input))
#
#     def _create_feedforward(self,
#                             model_kwargs):
#
#         act_fn_str = model_kwargs['ff_kwargs']['activation_str']
#         if act_fn_str == 'relu':
#             act_fn_constructor = nn.ReLU
#         elif act_fn_str == 'sigmoid':
#             act_fn_constructor = nn.Sigmoid
#         elif act_fn_str == 'tanh':
#             act_fn_constructor = nn.Tanh
#         else:
#             raise NotImplementedError
#
#         layer_widths = [self.input_size] + list(model_kwargs['ff_kwargs']['layer_widths'])
#         feedforward = OrderedDict()
#         for i in range(len(layer_widths) - 1):
#             feedforward[f'linear_{i}'] = nn.Linear(
#                 in_features=layer_widths[i],
#                 out_features=layer_widths[i + 1])
#             feedforward[f'{act_fn_str}_{i}'] = act_fn_constructor()
#         feedforward[f'linear{i + 1}'] = nn.Linear(
#             in_features=layer_widths[-1],
#             out_features=self.output_size)
#         feedforward = nn.Sequential(feedforward)
#         return feedforward
#
#     def forward(self, model_input):
#
#         layer_input = torch.cat(
#             [model_input['stimulus'],
#              model_input['reward'].reshape(-1, 1, 1)],
#             dim=2)
#
#         # make sure no gradients backpropagate through
#         layer_input = layer_input.detach()
#
#         for i, layer in enumerate(self.feedforward):
#             if i == len(self.feedforward) - 2:
#                 penultimate_layer = layer_input
#             layer_input = layer(layer_input)
#         feedforward_output = layer_input
#
#         # shape: (batch size, 1, output dim e.g. 2)
#         softmax_output = self.softmax(feedforward_output)
#
#         forward_output = dict(
#             feedforward_output=feedforward_output,
#             softmax_output=softmax_output,
#             core_hidden=penultimate_layer)
#
#         return forward_output


class RecurrentModel(nn.Module):

    def __init__(self,
                 model_str,
                 model_kwargs):

        super(RecurrentModel, self).__init__()
        self.model_str = model_str
        assert model_str in {'rnn', 'lstm', 'gru'}
        self.model_kwargs = model_kwargs
        self.input_size = model_kwargs['input_size']
        self.output_size = model_kwargs['output_size']

        # create and save core i.e. the recurrent operation
        self.core = self._create_core(
            model_str=model_str,
            model_kwargs=model_kwargs)

        masks = self._create_connectivity_masks(
            model_str=model_str,
            model_kwargs=model_kwargs)
        self.input_mask = masks['input_mask']
        self.recurrent_mask = masks['recurrent_mask']
        self.readout_mask = masks['readout_mask']

        self.description_str = create_description_str(model=self)

        self.core_hidden = None
        self.readout = nn.Linear(
            in_features=model_kwargs['core_kwargs']['hidden_size'],
            out_features=self.output_size,
            bias=True)

        # converts all weights into doubles i.e. float64
        # this prevents PyTorch from breaking when multiplying float32 * float64
        self.double()

        # TODO figure out why writing the model to tensorboard doesn't work
        # dummy_input = torch.zeros(size=(10, 1, 1), dtype=torch.double)
        # tensorboard_writer.add_graph(
        #     model=self,
        #     input_to_model=dict(stimulus=dummy_input))

    def _create_core(self, model_str, model_kwargs):
        if model_str == 'lstm':
            core_constructor = nn.LSTM
        elif model_str == 'rnn':
            core_constructor = nn.RNN
        elif model_str == 'gru':
            core_constructor = nn.GRU
        else:
            raise ValueError('Unknown core string')

        core = core_constructor(
            input_size=self.input_size,
            batch_first=True,
            **model_kwargs['core_kwargs'])

        param_init_str = model_kwargs['param_init']
        if param_init_str == 'default':
            return core
        elif param_init_str == 'eye':
            param_init_fn = init.eye_
        elif param_init_str == 'zeros':
            param_init_fn = init.zeros_
        elif param_init_str == 'ones':
            # TODO: breaks with error
            # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            param_init_fn = init.ones_
        elif param_init_str == 'uniform':
            param_init_fn = init.uniform
        elif param_init_str == 'normal':
            # TODO: breaks with error
            # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            param_init_fn = init.normal_
        elif param_init_str == 'xavier_uniform':
            param_init_fn = init.xavier_uniform_
        elif param_init_str == 'xavier_normal':
            param_init_fn = init.xavier_normal_
        else:
            raise NotImplementedError(f'Weight init function {param_init_str} unrecognized')

        if param_init_str != 'default':
            for weight in core.all_weights:
                for parameter in weight:
                    # some initialization functions e.g. eye only apply to 2D tensors
                    # skip the 1D tensors e.g. bias
                    try:
                        param_init_fn(parameter)
                    except ValueError:
                        continue

        return core

    def _create_connectivity_masks(self, model_str, model_kwargs):

        hidden_size = model_kwargs['core_kwargs']['hidden_size']

        # if mask not specifies, set to defaults
        for mask_str in ['input_mask', 'recurrent_mask', 'readout_mask']:
            if mask_str not in model_kwargs['connectivity_kwargs']:
                if mask_str == 'input_mask':
                    model_kwargs['connectivity_kwargs'][mask_str] = mask_str
                elif mask_str == 'readout_mask':
                    model_kwargs['connectivity_kwargs'][mask_str] = mask_str
                elif mask_str == 'recurrent_mask':
                    model_kwargs['connectivity_kwargs'][mask_str] = 'none'

        # determine how much to inflate
        if self.model_str == 'rnn':
            size_prefactor = 1
        elif self.model_str == 'gru':
            size_prefactor = 3
        elif self.model_str == 'lstm':
            size_prefactor = 4

        # create input-to-hidden, hidden-to-hidden, hidden-to-readout masks
        masks = dict()
        for mask_str, mask_type_str in model_kwargs['connectivity_kwargs'].items():

            if mask_str == 'input_mask':
                mask_shape = (size_prefactor * hidden_size, self.input_size)
            elif mask_str == 'recurrent_mask':
                mask_shape = (size_prefactor * hidden_size, hidden_size)
            elif mask_str == 'readout_mask':
                mask_shape = (self.output_size, hidden_size)
            else:
                raise ValueError(f'Unrecognized mask str: {mask_str}')

            mask = self._create_mask(
                mask_type_str=mask_type_str,
                output_shape=mask_shape[0],
                input_shape=mask_shape[1])

            masks[mask_str] = mask

        return masks

    def _create_mask(self, mask_type_str, output_shape, input_shape):

        if mask_type_str == 'none':
            connectivity_mask = np.ones(shape=(output_shape, input_shape))
        elif mask_type_str == 'input_mask':
            # special case for input - zeros except for first 30% of rows
            connectivity_mask = np.zeros(shape=(output_shape, input_shape))
            connectivity_mask[:int(0.3 * output_shape), :] = 1
        elif mask_type_str == 'readout_mask':
            # special case for output -
            connectivity_mask = np.zeros(shape=(output_shape, input_shape))
            connectivity_mask[:, -int(0.3 * input_shape):] = 1
        elif mask_type_str == 'diagonal':
            connectivity_mask = np.eye(N=output_shape, M=input_shape)
        elif mask_type_str == 'circulant':
            first_column = np.zeros(shape=output_shape)
            first_column[:int(0.2*output_shape)] = 1.
            connectivity_mask = scipy.linalg.circulant(c=first_column)
        elif mask_type_str == 'toeplitz':
            first_column = np.zeros(shape=output_shape)
            first_column[:int(0.2 * output_shape)] = 1.
            connectivity_mask = scipy.linalg.toeplitz(c=first_column)
        elif mask_type_str == 'small_world':
            graph = networkx.watts_strogatz_graph(
                n=output_shape,
                k=int(0.2 * output_shape),
                p=0.1)
            connectivity_mask = networkx.to_numpy_matrix(G=graph)
        elif mask_type_str.endswith('_block_diag'):
            # extract leading integer
            num_blocks = int(mask_type_str.split('_')[0])
            subblock_size = output_shape // num_blocks
            # check output size is exactly divisible by number of blocks
            assert num_blocks * subblock_size == output_shape
            connectivity_mask = scipy.linalg.block_diag(
                *[np.ones((subblock_size, subblock_size))] * num_blocks)
        else:
            raise ValueError(f'Unrecognized mask type str: {mask_type_str}')

        connectivity_mask = torch.from_numpy(connectivity_mask).double()
        return connectivity_mask

    def forward(self, model_input):
        """
        Performs a forward pass through model.


        :param model_input: dictionary containing 4 keys:
            stimulus: Tensor with shape (batch size, num steps, stimulus dimension)
            reward: Tensor with shape (batch size, 1 step)
            info: List of len batch size. Currently unused
            done: List of len batch size. Booleans indicating whether environment is done.
        :return forward_output: dictionary containing 4 keys:
            core_output: Tensor of shape (batch size, num steps, core dimension)
            core_hidden: Tensor of shape (batch size, num steps, core dimension)
            linear_output: Tensor of shape (batch size, num steps, output dimension)
            softmax_output: linear_output passed through softmax function
        """

        core_input = torch.cat(
            [model_input['stimulus'],
             model_input['reward'].reshape(-1, 1, 1)],
            dim=2)

        core_output, self.core_hidden = self.core(
            core_input,
            self.core_hidden)

        # hidden state is saved as (Number of RNN layers, Batch Size, Dimension)
        # swap so that hidden states is (Batch Size, Num of RNN Layers, Dimension)
        if self.model_str == 'rnn' or self.model_str == 'gru':
            core_hidden = self.core_hidden.transpose(0, 1)
        elif self.model_str == 'lstm':
            # hidden state is 2-tuple of (h_t, c_t). need to save both
            # stack h_t, c_t using last dimension
            # shape: (Batch Size, Num of RNN Layers, Dimension, 2)
            core_hidden = torch.stack(self.core_hidden, dim=-1).transpose(0, 1)
        else:
            raise NotImplementedError

        linear_output = self.readout(core_output)

        # shape: (batch size, 1, output dim e.g. 1)
        sigmoid_output = torch.sigmoid(linear_output)

        forward_output = dict(
            core_output=core_output,
            core_hidden=core_hidden,
            linear_output=linear_output,
            sigmoid_output=sigmoid_output)

        return forward_output

    def reset_core_hidden(self):
        self.core_hidden = None

    def apply_connectivity_masks(self):

        self.readout.weight.data[:] = torch.mul(
            self.readout.weight, self.readout_mask)

        # if self.model_str == 'rnn':
        self.core.weight_ih_l0.data[:] = torch.mul(
            self.core.weight_ih_l0, self.input_mask)
        self.core.weight_hh_l0.data[:] = torch.mul(
            self.core.weight_hh_l0, self.recurrent_mask)
        # elif self.model_str == 'lstm':
        #     raise NotImplementedError('LSTM masking not yet implemented')
        # elif self.model_str == 'gru':
        #     raise NotImplementedError('GRU masking not yet implemented')
        # else:
        #     raise NotImplementedError('Unrecognized Model String')


def create_description_str(model):
    description_str = '{}'.format(model.model_str)
    for key, value in model.model_kwargs.items():
        if key == 'input_size' or key == 'output_size':
            continue
        if isinstance(value, dict):
            for nkey, nvalue in value.items():
                description_str += ', {}={}'.format(str(nkey), str(nvalue))
        else:
            description_str += ', {}={}'.format(str(key), str(value))
    print(description_str)
    return description_str
