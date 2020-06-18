from collections import OrderedDict
import networkx
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init


class BayesianActor(object):

    def __init__(self):

        # to initialize, call self.reset()
        self.transition_probs = None
        self.emission_probs = None
        self.curr_block_posterior = None
        self.curr_stim_posterior = None
        self.mu = None
        self.prob_mu_given_stim_side = None
        self.start = None

    def reset(self,
              num_sessions,
              trials_per_block_param,
              block_side_probs,
              possible_trial_strengths,
              possible_trial_strengths_probs):

        self.emission_probs = np.array(
            block_side_probs)

        self.transition_probs = np.array([
            [1 - trials_per_block_param, trials_per_block_param],
            [trials_per_block_param, 1 - trials_per_block_param]])

        self.curr_stim_posterior = np.full(
            shape=(num_sessions, 1, 2),
            fill_value=0.5)
        self.curr_block_posterior = np.full(
            shape=(num_sessions, 1, 2),
            fill_value=0.5)

        mu = np.sort(np.concatenate(
            [np.array(possible_trial_strengths[1:]),
             -1. * np.array(possible_trial_strengths)]))
        prob_mu = possible_trial_strengths_probs

        # P(mu_n | s_n) as a matrix with shape (2 * number of stimulus strengths - 1, 2)
        # - 1 is for stimulus strength 0, which both stimulus sides can generate
        prob_mu_given_stim_side = np.zeros(shape=(len(mu), 2))
        prob_mu_given_stim_side[:len(prob_mu), 0] = prob_mu[::-1]
        prob_mu_given_stim_side[len(prob_mu) - 1:, 1] = prob_mu

        self.mu = mu
        self.prob_mu_given_stim_side = prob_mu_given_stim_side
        self.start = True

    def __call__(self, model_input):
        """
        Performs a forward pass through model.

        WARNING: HAS NOT BEEN TESTED ON BATCH SIZE > 1

        :param model_input: dictionary containing 4 keys:
            stimulus: Tensor with shape (batch size, 1 step, stimulus dimension)
            reward: Tensor with shape (batch size, 1 step)
            info: List of len batch size. Currently unused
            done: List of len batch size. Booleans indicating whether environment is done.
        :return forward_output: dictionary containing 4 keys:
            core_output: Tensor of shape (batch size, num steps, core dimension)
            core_hidden: Tensor of shape (batch size, num steps, core dimension)
            linear_output: Tensor of shape (batch size, num steps, output dimension)
            prob_output: Tensor of shape (batch size, num steps, output dimension)
        """

        # print('Stimulus Prior: ', self.curr_stim_posterior[0, 0])
        # print('Block Prior: ', self.curr_block_posterior[0, 0])

        # blank dt, skip
        if torch.all(model_input['stimulus'] == 0.) and torch.all(model_input['reward'] == 0.):
            # print('Blank dt, skipping')
            pass
        # trial end, update block posterior
        elif torch.all(model_input['stimulus'] == 0.) and torch.all(model_input['reward'] != 0.):
            # if reward was positive, trial side is the current stimulus posterior
            # otherwise, trial side is the opposite of the current stimulus posterior
            chosen_action = np.round(self.curr_stim_posterior)
            # shape = (batch size, # time steps = 1, block sides = 2)
            correct_stim_side = np.where(
                model_input['reward'].detach().numpy() == 1,
                chosen_action,
                np.abs(1 - chosen_action))
            correct_stim_side = np.argmax(correct_stim_side, axis=2)
            # print('Chosen action: ', chosen_action[0, 0, :])
            # print('Correct action: ', correct_stim_side[0, 0])
            self.update_block_posterior(correct_stim_side=correct_stim_side)
        # within trial, update stimulus posterior
        else:
            self.update_stim_posterior(
                stimulus=model_input['stimulus'].detach().numpy())

        # print('Stimulus Posterior: ', self.curr_stim_posterior[0, 0])
        # print('Block Posterior: ', self.curr_block_posterior[0, 0])

        # switch to PyTorch tensors for consistency with API
        model_output = dict(
            prob_output=torch.from_numpy(self.curr_stim_posterior),
            core_hidden=torch.from_numpy(self.curr_block_posterior),
            linear_output=torch.from_numpy(np.full_like(self.curr_stim_posterior, fill_value=np.nan)),
            core_output=torch.from_numpy(np.full_like(self.curr_block_posterior, fill_value=np.nan)))

        return model_output

    def update_stim_posterior(self, stimulus):

        # shape: (batch size, # time steps = 1, # of observations = 1)
        diff_obs = stimulus[:, :, 1, np.newaxis] - stimulus[:, :, 0, np.newaxis]
        # print('Diff of Obs: ', diff_obs[0, 0, :])

        # P(\mu_n, s_n | history) = P(\mu_n | s_n) P(s_n | history)
        # shape = (batch size, # time steps = 1,
        #          # of possible signed stimuli strengths, num stimulus sides)
        stim_side_strength_joint_prob = np.einsum(
            'us,bts->btus',
            self.prob_mu_given_stim_side,
            self.curr_stim_posterior)

        # P(o_t | \mu_n, s_n) , also = P(o_t | \mu_n)
        # shape = (batch size, # observations = 1,
        #          # of observations = 1, # of possible signed stimuli strengths)
        diff_obs_likelihood = scipy.stats.norm.pdf(
            np.expand_dims(diff_obs, axis=1),
            loc=self.mu,
            scale=np.sqrt(2) * np.ones_like(self.mu))  # scale is std dev

        # multiply likelihood by prior i.e. previous posterior
        # P(o_{<=t}, \mu_n, s_n | history) = P(o_{<=t} | \mu_n, s_n) P(\mu_n, s_n | history)
        # shape = (batch size, # of time steps = 1, # of observations = 1,
        #          # of possible signed stimuli strengths, # of trial sides i.e. 2)
        diff_obs_stim_side_strength_joint_prob = np.einsum(
            'btou,btus->btous',  # this may be wrong
            diff_obs_likelihood,
            stim_side_strength_joint_prob)
        assert len(diff_obs_stim_side_strength_joint_prob.shape) == 5

        # marginalize out mu_n (strength)
        # shape = (batch size, # of time steps = 1,
        #          # of observations, # of trial sides i.e. 2)
        diff_obs_stim_side_joint_prob = np.sum(
            diff_obs_stim_side_strength_joint_prob,
            axis=3)
        assert len(diff_obs_stim_side_joint_prob.shape) == 4

        # normalize by p(o_{<=t})
        # shape = (num of observations, # of time steps, # of obs = 1)
        diff_obs_marginal_prob = np.sum(
            diff_obs_stim_side_joint_prob,
            axis=3)
        assert len(diff_obs_marginal_prob.shape) == 3

        # shape = (num of observations, # of time steps,
        #          # of trial sides i.e. 2)
        curr_stim_posterior = np.divide(
            diff_obs_stim_side_joint_prob,
            np.expand_dims(diff_obs_marginal_prob, axis=1)  # expand to broadcast
        )[:, :, 0, :]
        assert len(curr_stim_posterior.shape) == 3
        assert np.allclose(
            np.sum(curr_stim_posterior, axis=2),
            np.ones(shape=curr_stim_posterior.shape[:-1]))

        self.curr_stim_posterior = curr_stim_posterior

    def update_block_posterior(self, correct_stim_side):
        """

        :param trial_side: either 0 (left) or 1 (right)
        :return:
        """
        # ensure integers to allow indexing
        correct_stim_side = correct_stim_side.astype(np.int)
        if self.start:
            # shape: (batch size, time steps = 1,
            curr_joint_prob = np.multiply(
                self.emission_probs[correct_stim_side],
                self.curr_block_posterior)
            self.start = False
        else:
            curr_joint_prob = np.multiply(
                self.emission_probs[correct_stim_side],
                np.einsum(
                    'ib,stb->sti',
                    self.transition_probs,
                    self.curr_block_posterior))

        # normalize to get P(b_n | s_{<=n})
        # np.sum(curr_joint_prob) is marginalizing over b_{n} i.e. \sum_{b_n} P(b_n, s_n |x_{<=n-1})
        self.curr_block_posterior = curr_joint_prob / np.sum(curr_joint_prob, axis=2)

        # create prior for next stimulus
        self.curr_stim_posterior = np.einsum(
            'ib,stb->sti',
            self.transition_probs,
            self.curr_block_posterior)


class ExponentialWeightedActor(object):

    def __init__(self):

        # to initialize, call self.reset()
        self.decay = None
        self.exp_decaying_stimulus_prior = None
        self.curr_stim_posterior = None
        self.mu = None
        self.prob_mu_given_stim_side = None
        self.start = None

    def reset(self,
              num_sessions,
              decay,
              possible_trial_strengths,
              possible_trial_strengths_probs):

        self.decay = decay
        self.exp_decaying_stimulus_prior = np.full(
            shape=(num_sessions, 1, 2),
            fill_value=0.5)

        self.curr_stim_posterior = np.full(
            shape=(num_sessions, 1, 2),
            fill_value=0.5)

        mu = np.sort(np.concatenate(
            [np.array(possible_trial_strengths[1:]),
             -1. * np.array(possible_trial_strengths)]))
        prob_mu = possible_trial_strengths_probs

        # P(mu_n | s_n) as a matrix with shape (2 * number of stimulus strengths - 1, 2)
        # - 1 is for stimulus strength 0, which both stimulus sides can generate
        prob_mu_given_stim_side = np.zeros(shape=(len(mu), 2))
        prob_mu_given_stim_side[:len(prob_mu), 0] = prob_mu[::-1]
        prob_mu_given_stim_side[len(prob_mu) - 1:, 1] = prob_mu

        self.mu = mu
        self.prob_mu_given_stim_side = prob_mu_given_stim_side
        self.start = True

    def __call__(self, model_input):
        """
        Performs a forward pass through model.

        WARNING: HAS NOT BEEN TESTED ON BATCH SIZE > 1

        :param model_input: dictionary containing 4 keys:
            stimulus: Tensor with shape (batch size, 1 step, stimulus dimension)
            reward: Tensor with shape (batch size, 1 step)
            info: List of len batch size. Currently unused
            done: List of len batch size. Booleans indicating whether environment is done.
        :return forward_output: dictionary containing 4 keys:
            core_output: Tensor of shape (batch size, num steps, core dimension)
            core_hidden: Tensor of shape (batch size, num steps, core dimension)
            linear_output: Tensor of shape (batch size, num steps, output dimension)
            prob_output: Tensor of shape (batch size, num steps, output dimension)
        """

        # print('Stimulus Prior: ', self.curr_stim_posterior[0, 0])
        # print('Block Prior: ', self.curr_block_posterior[0, 0])

        # blank dt, skip
        if torch.all(model_input['stimulus'] == 0.) and torch.all(model_input['reward'] == 0.):
            # print('Blank dt, skipping')
            pass
        # trial end, update block posterior
        elif torch.all(model_input['stimulus'] == 0.) and torch.all(model_input['reward'] != 0.):
            # if reward was positive, trial side is the current stimulus posterior
            # otherwise, trial side is the opposite of the current stimulus posterior
            chosen_action = np.round(self.curr_stim_posterior)
            # shape = (batch size, # time steps = 1, block sides = 2)
            correct_stim_side = np.where(
                model_input['reward'].detach().numpy() == 1,
                chosen_action,
                np.abs(1 - chosen_action))
            correct_stim_side = np.argmax(correct_stim_side, axis=2)
            # print('Chosen action: ', chosen_action[0, 0, :])
            # print('Correct action: ', correct_stim_side[0, 0])
            self.update_stim_prior(correct_stim_side=correct_stim_side)
        # within trial, update stimulus posterior
        else:
            self.update_stim_posterior(
                stimulus=model_input['stimulus'].detach().numpy())

        # print('Stimulus Posterior: ', self.curr_stim_posterior[0, 0])
        # print('Block Posterior: ', self.curr_block_posterior[0, 0])

        # switch to PyTorch tensors for consistency with API
        model_output = dict(
            prob_output=torch.from_numpy(self.curr_stim_posterior),
            core_hidden=torch.from_numpy(self.exp_decaying_stimulus_prior),
            linear_output=torch.from_numpy(np.full_like(self.curr_stim_posterior, fill_value=np.nan)),
            core_output=torch.from_numpy(np.full_like(self.exp_decaying_stimulus_prior, fill_value=np.nan)))

        return model_output

    def update_stim_posterior(self, stimulus):

        # shape: (batch size, # time steps = 1, # of observations = 1)
        diff_obs = stimulus[:, :, 1, np.newaxis] - stimulus[:, :, 0, np.newaxis]
        # print('Diff of Obs: ', diff_obs[0, 0, :])

        # P(\mu_n, s_n | history) = P(\mu_n | s_n) P(s_n | history)
        # shape = (batch size, # time steps = 1,
        #          # of possible signed stimuli strengths, num stimulus sides)
        stim_side_strength_joint_prob = np.einsum(
            'us,bts->btus',
            self.prob_mu_given_stim_side,
            self.curr_stim_posterior)

        # P(o_t | \mu_n, s_n) , also = P(o_t | \mu_n)
        # shape = (batch size, # observations = 1,
        #          # of observations = 1, # of possible signed stimuli strengths)
        diff_obs_likelihood = scipy.stats.norm.pdf(
            np.expand_dims(diff_obs, axis=1),
            loc=self.mu,
            scale=np.sqrt(2) * np.ones_like(self.mu))  # scale is std dev

        # multiply likelihood by prior i.e. previous posterior
        # P(o_{<=t}, \mu_n, s_n | history) = P(o_{<=t} | \mu_n, s_n) P(\mu_n, s_n | history)
        # shape = (batch size, # of time steps = 1, # of observations = 1,
        #          # of possible signed stimuli strengths, # of trial sides i.e. 2)
        diff_obs_stim_side_strength_joint_prob = np.einsum(
            'btou,btus->btous',  # this may be wrong
            diff_obs_likelihood,
            stim_side_strength_joint_prob)
        assert len(diff_obs_stim_side_strength_joint_prob.shape) == 5

        # marginalize out mu_n (strength)
        # shape = (batch size, # of time steps = 1,
        #          # of observations, # of trial sides i.e. 2)
        diff_obs_stim_side_joint_prob = np.sum(
            diff_obs_stim_side_strength_joint_prob,
            axis=3)
        assert len(diff_obs_stim_side_joint_prob.shape) == 4

        # normalize by p(o_{<=t})
        # shape = (num of observations, # of time steps, # of obs = 1)
        diff_obs_marginal_prob = np.sum(
            diff_obs_stim_side_joint_prob,
            axis=3)
        assert len(diff_obs_marginal_prob.shape) == 3

        # shape = (num of observations, # of time steps,
        #          # of trial sides i.e. 2)
        curr_stim_posterior = np.divide(
            diff_obs_stim_side_joint_prob,
            np.expand_dims(diff_obs_marginal_prob, axis=1)  # expand to broadcast
        )[:, :, 0, :]
        assert len(curr_stim_posterior.shape) == 3
        assert np.allclose(
            np.sum(curr_stim_posterior, axis=2),
            np.ones(shape=curr_stim_posterior.shape[:-1]))

        self.curr_stim_posterior = curr_stim_posterior

    def update_stim_prior(self, correct_stim_side):
        """

        :param trial_side: either 0 (left) or 1 (right)
        :return:
        """
        # ensure integers to allow indexing
        correct_stim_side = correct_stim_side.astype(np.int)

        # the stimulus prior should be between [0, 1], but we want decay
        # to fall to 0.5, not 0. So variable tranform, decay, then
        # transform back

        self.exp_decaying_stimulus_prior *= self.decay
        # TODO: check that this slicing is correct
        self.exp_decaying_stimulus_prior[:, :, correct_stim_side] += (1. - self.decay)

        # check that this is valid probability distribution
        assert np.all(np.isclose(self.exp_decaying_stimulus_prior.sum(axis=2), 1.))

        # create prior for next stimulus
        self.curr_stim_posterior = self.exp_decaying_stimulus_prior


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
                 model_architecture,
                 model_kwargs):

        super(RecurrentModel, self).__init__()
        self.model_str = model_architecture
        assert model_architecture in {'rnn', 'lstm', 'gru'}
        self.model_kwargs = model_kwargs
        self.input_size = model_kwargs['input_size']
        self.output_size = model_kwargs['output_size']

        # create and save core i.e. the recurrent operation
        self.core = self._create_core(
            model_architecture=model_architecture,
            model_kwargs=model_kwargs)

        masks = self._create_connectivity_masks(
            model_str=model_architecture,
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

        if self.output_size == 1:
            self.prob_fn = nn.Sigmoid()
        elif self.output_size == 2:
            self.prob_fn = nn.Softmax(dim=2)

        # converts all weights into doubles i.e. float64
        # this prevents PyTorch from breaking when multiplying float32 * float64
        self.double()

        # TODO figure out why writing the model to tensorboard doesn't work
        # dummy_input = torch.zeros(size=(10, 1, 1), dtype=torch.double)
        # tensorboard_writer.add_graph(
        #     model=self,
        #     input_to_model=dict(stimulus=dummy_input))

    def _create_core(self, model_architecture, model_kwargs):
        if model_architecture == 'lstm':
            core_constructor = nn.LSTM
        elif model_architecture == 'rnn':
            core_constructor = nn.RNN
        elif model_architecture == 'gru':
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
            first_column[:int(0.2 * output_shape)] = 1.
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
            stimulus: Tensor with shape (batch size, 1 step, stimulus dimension)
            reward: Tensor with shape (batch size, 1 step)
            info: List of len batch size. Currently unused
            done: List of len batch size. Booleans indicating whether environment is done.
        :return forward_output: dictionary containing 4 keys:
            core_output: Tensor of shape (batch size, num steps, core dimension)
            core_hidden: Tensor of shape (batch size, num steps, core dimension)
            linear_output: Tensor of shape (batch size, num steps, output dimension)
            prob_output: Tensor of shape (batch size, num steps, output dimension)
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
        prob_output = self.prob_fn(linear_output)

        # if probability function is sigmoid, add 1 - output to get 2D distribution
        if self.output_size == 1:
            prob_output = torch.cat([1 - prob_output, prob_output], dim=2)
            # TODO: implement linear output i.e. inverse sigmoid
            linear_output = None
            raise NotImplementedError

        forward_output = dict(
            core_output=core_output,
            core_hidden=core_hidden,
            linear_output=linear_output,
            prob_output=prob_output)

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
