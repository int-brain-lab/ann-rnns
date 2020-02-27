import numpy as np
import pandas as pd
import torch

from utils.models import FeedforwardModel, RecurrentModel


def create_model(model_str=None,
                 model_kwargs=None):

    # defaults
    if model_str is None:
        model_str = 'rnn'
    if model_kwargs is None:
        model_kwargs = dict(
            input_size=2,
            output_size=2,
            core_kwargs=dict(
                num_layers=1,
                hidden_size=50),
            param_init='default',
            connectivity_kwargs=dict(
                recurrent_mask='small_world',
            ))

    # model = create_model(
    #     model_str='ff',
    #     model_kwargs=dict(ff_kwargs=dict(activation_str='relu',
    #                                      layer_widths=[10, 10])))

    if model_str in {'rnn', 'lstm', 'gru'}:
        model = RecurrentModel(
            model_str=model_str,
            model_kwargs=model_kwargs)
    elif model_str in {'ff'}:
        model = FeedforwardModel(
            model_str=model_str,
            model_kwargs=model_kwargs)
    else:
        raise NotImplementedError(f'Unknown core_str: {model_str}')

    return model


def create_optimizer(model,
                     optimizer_str='sgd',
                     optimizer_kwargs=None,
                     lr=0.01):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if optimizer_str == 'sgd':
        optimizer_constructor = torch.optim.SGD
    elif optimizer_str == 'adam':
        optimizer_constructor = torch.optim.Adam
    elif optimizer_str == 'rmsprop':
        optimizer_constructor = torch.optim.RMSprop
    else:
        raise NotImplementedError('Unknown optimizer string')

    optimizer = optimizer_constructor(
        params=model.parameters(),
        lr=lr,
        **optimizer_kwargs)

    return optimizer


def load_checkpoint(checkpoint_path,
                    tensorboard_writer):

    save_dict = torch.load(checkpoint_path)

    model = create_model(model_str=save_dict['model_str'],
                         model_kwargs=save_dict['model_kwargs'])
    model.load_state_dict(save_dict['model_state_dict'])

    optimizer = create_optimizer(
        model=model,
        optimizer_str='sgd')
    optimizer.load_state_dict(save_dict['optimizer_state_dict'])

    global_step = save_dict['global_step']

    return model, optimizer, global_step


def run_envs(model,
             envs):

    total_reward = torch.zeros(1, dtype=torch.double, requires_grad=True)
    total_loss = torch.zeros(1, dtype=torch.double, requires_grad=True)
    if hasattr(model, 'reset_core_hidden'):
        model.reset_core_hidden()  # reset core's hidden state
    step_output = envs.reset()

    # step until the first env finishes
    total_trials = 0
    while not np.any(step_output['done']):
        total_reward = total_reward + torch.sum(step_output['reward'])  # cannot use +=
        total_loss = total_loss + torch.sum(step_output['loss'])

        model_output = model(step_output)

        # squeeze to remove the timestep (i.e. middle dimension) for the environment
        step_output = envs.step(
            actions=model_output['softmax_output'].squeeze(1),
            core_hidden=model_output['core_hidden'])

        total_trials += 1

    # divide by total trials, batch size
    avg_reward = total_reward / (total_trials * len(envs))
    avg_loss = total_loss / (total_trials * len(envs))

    # construct output dictionary
    # most of the environments will be truncated early because we stop as soon
    # as the first environment finishes. we detect the indices where this happens
    # and exclude them
    # unused trials are identified by the property that both the action
    # probabilities are 0. This is due to the arrays being initialized to 0.
    actions_probs = np.concatenate(
        [env.actions.detach().numpy() for env in envs])
    used_trial_indices = np.logical_and(
        actions_probs[:, 0] != 0.,
        actions_probs[:, 1] != 0.)
    actions_probs = actions_probs[used_trial_indices]
    actions_chosen = np.argmax(actions_probs, axis=1)

    stimuli = np.concatenate(
        [env.stimuli.detach().numpy() for env in envs])[used_trial_indices]
    stimuli_sides = np.concatenate(
        [env.stimuli_sides.detach().numpy() for env in envs])[used_trial_indices]
    stimuli_sides_indices = (1. + stimuli_sides) / 2
    stimuli_strengths = np.concatenate(
        [env.stimuli_strengths.detach().numpy() for env in envs])[used_trial_indices]
    stimuli_preferred_sides = np.concatenate(
        [env.stimuli_preferred_sides.detach().numpy() for env in envs])[used_trial_indices]
    rewards = np.concatenate(
        [env.rewards.detach().numpy() for env in envs])[used_trial_indices]
    losses = np.concatenate(
        [env.losses.detach().numpy() for env in envs])[used_trial_indices]
    stimuli_block_number = np.concatenate(
        [env.stimuli_block_number for env in envs])[used_trial_indices]
    trial_num_within_block = np.concatenate(
        [env.trial_num_within_block for env in envs])[used_trial_indices]

    actions_correct = np.equal(actions_chosen, stimuli_sides_indices)
    avg_correct_choice = np.mean(actions_correct)

    model_correct_action_probs = actions_probs[
        np.arange(len(actions_probs)),
        (stimuli_sides + 1) // 2]

    model_hidden_states = np.concatenate([
        np.stack(env.model_hidden_states)
        for env in envs])

    env_num = np.concatenate([
        np.full(fill_value=i, shape=env.total_num_trials)
        for i, env in enumerate(envs)])[used_trial_indices]

    trial_data = pd.DataFrame(dict(
        env_num=env_num,
        stimuli=stimuli,
        stimuli_sides=stimuli_sides,
        stimuli_strengths=stimuli_strengths,
        stimuli_preferred_sides=stimuli_preferred_sides,
        stimuli_block_number=stimuli_block_number,
        rewards=rewards,
        losses=losses,
        actions_chosen=actions_chosen,
        actions_correct=actions_correct,
        model_left_action_probs=actions_probs[:, 0],
        model_right_action_probs=actions_probs[:, 1],
        model_correct_action_probs=model_correct_action_probs,
        trial_num_within_block=trial_num_within_block,
    ))

    run_envs_output = dict(
        trial_data=trial_data,
        hidden_states=model_hidden_states,
        avg_reward=avg_reward,
        avg_correct_choice=avg_correct_choice,
        avg_loss=avg_loss,
    )

    return run_envs_output


def save_train_output(test_or_train_output):
    raise NotImplementedError
