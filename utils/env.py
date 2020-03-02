import pandas as pd
import gym
from gym import spaces
import numpy as np
import torch
from torch.nn import NLLLoss

from utils.stimuli import VectorStimulusCreator
# from utils.subproc_env import SubprocVecEnv
from utils.vec_env import VecEnv


class IBLSession(gym.Env):

    def __init__(self,
                 stimulus_creator=VectorStimulusCreator(),
                 block_side_probs=((0.8, 0.2), (0.2, 0.8)),
                 loss_fn_str='nll',
                 blocks_per_session=10,
                 trials_per_block_param=1 / 60,
                 min_trials_per_block=60,
                 max_trials_per_block=100,
                 max_rnn_steps_per_trial=10,
                 time_delay_penalty=0.05):

        """

        :param stimulus_creator: object that must have
            observation_space: type gym.spaces
            create_block_stimuli(): method that generates stimuli
        :param loss_fn_str:
        :param blocks_per_session:
        :param trials_per_block_param:
        :param max_rnn_steps_per_trial:
        """

        self._check_stimulus_creator(
            stimulus_creator=stimulus_creator)
        self.stimulus_creator = stimulus_creator

        # probability of this block
        self.block_side_probs = block_side_probs

        self.blocks_per_session = blocks_per_session
        self.min_trials_per_block = min_trials_per_block
        self.max_trials_per_block = max_trials_per_block
        self.trials_per_block_param = trials_per_block_param
        self.max_rnn_steps_per_trial = max_rnn_steps_per_trial
        self.action_space = spaces.Discrete(2)  # left or right
        self.observation_space = stimulus_creator.observation_space
        self.reward_range = (0, 1)
        self.loss_fn_str = loss_fn_str
        self.time_delay_penalty = time_delay_penalty
        self.loss_fn = self.create_loss_fn(
            loss_fn_str=loss_fn_str,
            time_delay_penalty=time_delay_penalty)
        self.reward_fn = self.create_reward_fn(
            time_delay_penalty=time_delay_penalty)

        # to (re)initialize the following variables, call self.reset()
        self.num_trials_per_block = None
        self.session_data = None
        self.stimuli = None
        self.trial_stimulus_side = None
        self.block_stimulus_side = None
        self.losses = None
        self.current_trial_within_block = None
        self.current_block_within_session = None
        self.current_rnn_step_within_trial = None
        self.current_trial = None
        self.current_rnn_step = None
        self.total_num_rnn_steps = None

    @staticmethod
    def _check_stimulus_creator(stimulus_creator):

        # check that stimulus_creator is valid object
        assert stimulus_creator is not None
        assert hasattr(stimulus_creator, 'create_block_stimuli')
        assert hasattr(stimulus_creator, 'observation_space')

    @staticmethod
    def create_loss_fn(loss_fn_str, time_delay_penalty):
        """

        :param loss_fn_str: str specifying the desired loss function
        :return: loss_fn:   must have two keyword arguments, target and input
                            target.shape should be (batch size = 1,)
                            input.shape should be (batch size = 1, num actions = 2,)
        """

        if loss_fn_str == 'nll':
            nllloss = NLLLoss()

            def loss_fn(target, input):
                # TODO: adding time delay penalty is currently pointless
                loss = nllloss(target=target, input=input) + time_delay_penalty
                return loss

        else:
            raise NotImplementedError

        return loss_fn

    @staticmethod
    def create_reward_fn(time_delay_penalty):
        """
        :return: reward_fn:   must have two keyword arguments, target and input
                    target.shape should be (batch size = 1,)
                    input.shape should be (batch size = 1, num actions = 2,)
        """

        def reward_fn(target, input):
            # max returns (value, index)
            max_prob_idx = torch.max(input, dim=1)[1]

            # for an action to be rewarded, the model must have made the correct choice
            reward = (target == max_prob_idx).double()

            return reward

        return reward_fn

    def close(self, env_index):

        # add an indicator of which dataframe corresponds to which environment
        self.session_data['env_index'] = env_index

        # truncate unused rows
        self.session_data.drop(
            np.arange(self.current_rnn_step, len(self.session_data)),
            inplace=True)

    def reset(self):
        """
        (Re)initializes experiment.

        Previously, we relied on the fact that the RNN had a fixed number of
        steps per trial to preallocate the entire session, and then iterated
        over the preallocated values. This is no longer the case because the
        model can now decide to act faster or slower.
        """

        # sample number of trials per block, ensuring values are between
        # [min_trials_per_block, max_trials_per_block]
        self.num_trials_per_block = [np.random.geometric(p=self.trials_per_block_param)
                                     for _ in range(self.blocks_per_session)]
        for i in range(len(self.num_trials_per_block)):
            num_trials = self.num_trials_per_block[i]
            if num_trials < self.min_trials_per_block:
                self.num_trials_per_block[i] = self.min_trials_per_block
            if num_trials > self.max_trials_per_block:
                self.num_trials_per_block[i] = self.max_trials_per_block

        # create Pandas DataFrame for tracking all session data
        max_rnn_steps_per_session = np.sum(self.num_trials_per_block) * self.max_rnn_steps_per_trial
        index = np.arange(max_rnn_steps_per_session)
        columns = ['rnn_step_within_session',
                   'block_index', 'trial_index', 'rnn_step_index',
                   'left_stimulus', 'right_stimulus', 'trial_stimulus_side',
                   'block_stimulus_side', 'loss', 'reward',
                   'left_action_prob', 'right_action_prob',
                   'hidden_state']
        self.session_data = pd.DataFrame(
            np.nan,  # initialize all to nan
            index=index,
            columns=columns,
            dtype=np.float16)

        # enable storing hidden states in the dataframe.
        # need to make the column have type object to doing so possible
        self.session_data.hidden_state = self.session_data.hidden_state.astype(object)

        self.current_trial_within_block = 0
        self.current_block_within_session = 0
        self.current_rnn_step_within_trial = 0
        self.current_trial = 0
        self.current_rnn_step = 0
        self.total_num_rnn_steps = 0

        # choose first block bias with 50-50 probability
        # TODO: figure out what to do with stimuli strengths
        current_block_side = np.random.choice([0, 1])
        stimuli, trial_stimulus_side, stimuli_strengths = [], [], []
        block_stimulus_side = []
        for num_trials in self.num_trials_per_block:
            stimulus_creator_output = self.stimulus_creator.create_block_stimuli(
                num_trials=num_trials,
                block_side_bias_probabilities=self.block_side_probs[current_block_side])
            stimuli.append(stimulus_creator_output['stimuli'])
            trial_stimulus_side.append(stimulus_creator_output['sampled_sides'])
            block_stimulus_side.append(-1 if current_block_side == 0 else 1)
            current_block_side = 1 if current_block_side == 0 else 0

        # flatten each list of numpy arrays and convert to torch tensors
        self.stimuli = torch.from_numpy(np.concatenate(stimuli))
        self.trial_stimulus_side = torch.from_numpy(np.concatenate(trial_stimulus_side))
        self.block_stimulus_side = np.array(block_stimulus_side)
        self.losses = torch.zeros(max_rnn_steps_per_session)

        # create first observation
        step_output = dict(
            stimulus=self.stimuli[self.current_trial].reshape((1, -1)),
            reward=torch.zeros(1).double().requires_grad_(True),
            loss=torch.zeros(1).double().requires_grad_(True),
            info=None,
            done=True if self.current_block_within_session == self.blocks_per_session else False)

        return step_output

    def step(self,
             model_softmax_output,
             model_hidden):
        """

        :param model_softmax_output: shape (time step=1, 2)
        :param model_hidden:
        :return:
        """

        left_action_prob = model_softmax_output[0, 0].item()
        right_action_prob = model_softmax_output[0, 1].item()
        correct_action_index = ((self.trial_stimulus_side[self.current_trial].reshape((1,)) + 1) // 2)
        correct_action_prob = model_softmax_output[0, correct_action_index.item()].item()

        # target has shape (batch=1,)
        # reshape action to (batch=1, 2) since loss fn has no notion of sequence
        loss = self.loss_fn(
            target=correct_action_index,
            input=model_softmax_output)
        self.losses[self.current_rnn_step] = loss

        reward = self.reward_fn(
            target=correct_action_index,
            input=model_softmax_output)

        # record data
        timestep_data = dict(
            block_index=self.current_block_within_session,
            trial_index=self.current_trial_within_block,
            rnn_step_index=self.current_rnn_step_within_trial,
            rnn_step_within_session=self.current_rnn_step,
            left_stimulus=self.stimuli[self.current_trial_within_block][0].numpy(),
            right_stimulus=self.stimuli[self.current_trial_within_block][1].numpy(),
            trial_stimulus_side=self.trial_stimulus_side[self.current_trial].item(),
            block_stimulus_side=self.block_stimulus_side[self.current_trial].item(),
            loss=loss.item(),
            reward=reward.item(),
            left_action_prob=left_action_prob,
            right_action_prob=right_action_prob,
            correct_action_prob=correct_action_prob,
            hidden_state=model_hidden.detach().numpy())

        for column, value in timestep_data.items():
            self.session_data.at[self.current_rnn_step, column] = value

        # increment counters
        # current rnn step always advances
        self.current_rnn_step += 1

        # advance current rnn step within trial.
        self.current_rnn_step_within_trial += 1

        # max returns (value, index)
        max_prob = torch.max(model_softmax_output, dim=1)[0]
        # move to next trial if action made or if maxed out number of rnn_steps within trial
        if max_prob.item() > 0.9 or self.current_rnn_step_within_trial == self.max_rnn_steps_per_trial:

            self.current_rnn_step_within_trial = 0
            self.current_trial_within_block += 1

            # move to next block if finished trials within block
            if self.current_trial_within_block == self.num_trials_per_block[self.current_block_within_session]:
                self.current_block_within_session += 1
                self.current_rnn_step_within_trial = 0
                self.current_trial_within_block = 0

        stimulus = self.stimuli[self.current_trial_within_block]

        # store any additional desired information
        info = dict()

        # loss is used by the optimizer
        # reward is (possibly different) input to the model at the next time step
        step_output = dict(
            loss=loss,
            stimulus=stimulus.reshape((1, -1)),
            reward=reward,
            info=info,
            done=True if self.current_block_within_session == self.blocks_per_session else False)

        return step_output


def create_envs(kwargs,
                num_envs):
    def make_env(kwargs):
        def _f():
            return IBLSession(**kwargs)

        return _f

    envs = VecEnv(make_env_fn=make_env(kwargs=kwargs), num_env=num_envs)
    return envs


def create_training_choice_world(batch_size):
    """
    "training choice world during which visual stimuli have to be actively moved
    by the mouse; left and right stimuli are presented with equal probability."
    """

    kwargs = dict(
        blocks_per_session=10,
        stimulus_creator=VectorStimulusCreator())
    training_choice_worlds = [IBLSession(**kwargs)
                              for _ in range(batch_size)]
    return training_choice_worlds


def create_biased_choice_worlds(num_envs=11):
    """
    "biased choice world during which visual stimuli have to be actively moved
    by the mouse; left and right stimuli are presented with different probability
    in blocks of trials."
    """

    kwargs = dict(
        blocks_per_session=3,
        min_trials_per_block=10,
        max_trials_per_block=30,
        stimulus_creator=VectorStimulusCreator())
    envs = create_envs(kwargs=kwargs, num_envs=num_envs)
    return envs


def create_custom_worlds(tensorboard_writer,
                         num_envs=1,
                         blocks_per_session=3):
    kwargs = dict(
        blocks_per_session=blocks_per_session,
        stimulus_creator=VectorStimulusCreator())
    envs = create_envs(kwargs=kwargs, num_envs=num_envs)
    return envs
