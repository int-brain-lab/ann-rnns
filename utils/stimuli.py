import numpy as np
# from psychopy.visual.grating import GratingStim
from scipy.stats import truncnorm


def create_block_stimuli(num_trials,
                         block_side_bias_probabilities,
                         possible_trial_strengths,
                         possible_trial_strengths_probs,
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
        size=(num_trials, 1))
    signal_sides_indices = np.repeat(
        signal_sides_indices,
        axis=-1,
        repeats=max_rnn_steps_per_trial)

    trial_sides = 2*signal_sides_indices - 1

    # each trial strength will either be easy or hard
    # easy means that its signal distribution is drawn with mean 1.5
    # hard means that its signal distribution is drawn with mean 0.5
    trial_strengths = np.random.choice(
        possible_trial_strengths,
        p=possible_trial_strengths_probs,
        size=(num_trials, 1))

    # hold trial strength constant for duration of trial
    trial_strengths = np.repeat(
        a=trial_strengths,
        repeats=max_rnn_steps_per_trial,
        axis=1)

    signal = np.random.normal(
        loc=trial_strengths,
        scale=np.ones_like(trial_strengths))

    # add signal to noise
    # rely on nice identity matrix trick for converting boolean signal_side_indices
    # to one-hot encoded for indexing
    sampled_stimuli[np.eye(2)[signal_sides_indices].astype(bool)] = signal.flatten()

    # import matplotlib.pyplot as plt
    # plt.scatter(sampled_stimuli[:,:,0].flatten(), sampled_stimuli[:,:,1].flatten())
    # plt.show()

    output = dict(
        stimuli=sampled_stimuli,
        stimuli_strengths=trial_strengths,
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
