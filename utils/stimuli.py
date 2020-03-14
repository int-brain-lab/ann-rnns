from gym import spaces
import numpy as np
# from psychopy.visual.grating import GratingStim
from scipy.stats import truncnorm


# TODO: convert functions to classes that store outcomes and outcome probabilities

class StimulusCreator(object):

    def __init__(self,
                 stimulus_strengths,
                 stimulus_strength_probs,
                 observation_space):
        """

        :param stimulus_strengths:      Iterable of possible values defining how
                                        easily
        :param stimulus_strength_probs: probability vector, defining probability
                                        of each stimulus strength
        :param observation_space:       gym.space
        """

        self.stimulus_strengths = stimulus_strengths
        self.stimulus_strength_probs = stimulus_strength_probs
        self.observation_space = observation_space

    def create_block_stimuli(self,
                             num_trials,
                             block_side_bias_probabilities,
                             max_rnn_steps_per_trial):

        raise NotImplementedError


class VectorStimulusCreator(StimulusCreator):

    def __init__(self,
                 stimulus_strengths=None,
                 stimulus_strength_probs=None,
                 observation_space=None):
        
        # defaults
        if stimulus_strengths is None:
            stimulus_strengths = [1.]
        if stimulus_strength_probs is None:
            unif_prob = 1. / len(stimulus_strengths)
            stimulus_strength_probs = [unif_prob for _ in range(len(stimulus_strengths))]
        if observation_space is None:
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

        super(VectorStimulusCreator, self).__init__(
            stimulus_strengths=stimulus_strengths,
            stimulus_strength_probs=stimulus_strength_probs,
            observation_space=observation_space)

    def create_block_stimuli(self,
                             num_trials,
                             block_side_bias_probabilities,
                             max_rnn_steps_per_trial):

        # sample standard normal noise for both left and right stimuli
        sampled_stimuli = np.random.normal(
            loc=0,
            scale=0.1,
            size=(num_trials, max_rnn_steps_per_trial, 2))

        # now, determine which sides will have signal
        # -1 is left, +1 is right
        # these values also control the means of the distributions
        signal_sides_indices = np.random.choice(
            [0, 1],
            p=block_side_bias_probabilities,
            size=(num_trials, max_rnn_steps_per_trial))

        trial_sides = 2*signal_sides_indices - 1

        # each trial strength will either be easy or hard
        # easy means that its signal distribution is drawn with mean 1.5
        # hard means that its signal distribution is drawn with mean 0.5
        stimuli_strengths = np.random.choice(
            [1.5, 0.5],
            size=(num_trials, 1))

        # hold trial strength constant for duration of trial
        stimuli_strengths = np.repeat(
            a=stimuli_strengths,
            repeats=max_rnn_steps_per_trial,
            axis=1)

        signal = np.random.normal(
            loc=stimuli_strengths,
            scale=np.ones_like(stimuli_strengths))

        # add signal to noise
        # rely on nice identity matrix trick for converting boolean signal_side_indices
        # to one-hot encoded for indexing
        # signal_sides_indices = (signal_sides+1)//2
        sampled_stimuli[np.eye(2)[signal_sides_indices].astype(bool)] += signal.flatten()

        output = dict(
            stimuli=sampled_stimuli,
            stimuli_strengths=stimuli_strengths,
            trial_sides=trial_sides)

        return output


# class GratingCreator(StimulusCreator):
#
# TODO: fix this to create Gabor patches as specified
#
#     def __init__(self,
#                  stimulus_strengths=None,
#                  stimulus_strength_probs=None):
#
#         # defaults
#         if stimulus_strengths is None:
#             stimulus_strengths = [1, 0.5, 0.25, 0.125, 0.06, 0]
#         if stimulus_strength_probs is None:
#             stimulus_strength_probs = [2 / 11, 2 / 11, 2 / 11, 2 / 11, 2 / 11, 1 / 11]
#
#         super(GratingCreator, self).__init__(
#             stimulus_strengths=stimulus_strengths,
#             stimulus_strength_probs=stimulus_strength_probs)
#
#     def create_block_stimuli(self,
#                              block_num_trials,
#                              block_side_bias_probabilities):
#
#         sampled_strength = np.random.choice(
#             self.stimulus_strengths,
#             p=self.stimulus_strength_probs)
#
#
#         if side == 'left':
#             return GratingStim(tex='sin', mask='gauss')
#         elif side == 'right':
#             return GratingStim(tex='sin', mask='gauss')
#         else:
#             raise ValueError('Impermissible side: ', side)
