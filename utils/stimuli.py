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
                             block_num_trials,
                             block_side_bias_probabilities):

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
                             block_num_trials,
                             block_side_bias_probabilities):

        sampled_sides = np.random.choice(
            ['left', 'right'],
            p=block_side_bias_probabilities,
            size=block_num_trials)

        # convert left to -1, right to 1
        # these values control the means of the distributions
        sampled_sides = np.where(sampled_sides == 'left', -1, 1)

        sampled_strengths = np.random.choice(
            self.stimulus_strengths,
            p=self.stimulus_strength_probs,
            size=block_num_trials)

        std_dev = np.sqrt(1. / sampled_strengths)
        # sampled_noise = np.random.normal(
        #     loc=np.zeros_like(std_dev),
        #     scale=std_dev)
        sampled_noise = truncnorm.rvs(
            a=-1.5,
            b=1.5,
            loc=np.zeros_like(std_dev),
            scale=std_dev)

        # move means of sampled noise
        sampled_stimuli = sampled_sides + sampled_noise

        output = dict(
            sampled_stimuli=sampled_stimuli,
            sampled_sides=sampled_sides,
            sampled_strengths=sampled_strengths)

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
