import json
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch

from utils.models import RecurrentModel


def convert_session_data_to_ibl_changepoint_csv(session_data,
                                                env_block_side_probs,
                                                log_dir):

    trial_end_data = session_data[session_data.trial_end == 1.]
    logging.info(f'Total number of trials: {len(trial_end_data)}')

    trial_num = trial_end_data.trial_within_session.values.astype(np.int) + 1
    session_num = trial_end_data.session_index.values + 1

    # probabilities for block
    block_side_probs = np.array(env_block_side_probs)[:, 0]
    stim_probability_left = block_side_probs[((trial_end_data.block_side.values + 1)/2).astype(np.int)]

    # should be either 0, 0.0625, 0.125, 0.25, 1
    contrast = trial_end_data.trial_strength.values
    contrast_map = {
        0:      0,
        0.5:    0.0625,
        0.75:   0.125,
        1.0:    0.25,
        1.5:    1.}
    for old_contrast, new_contrast in contrast_map.items():
        contrast[contrast == old_contrast] = new_contrast

    # 35 is legacy number
    position = 35 * trial_end_data.trial_side.values
    response_choice = trial_end_data.action_side.values
    trial_correct = trial_end_data.correct_action_taken.values
    reaction_time = trial_end_data.rnn_step_index.values

    # assert all correct actions have position sign matching response choice sign
    assert np.all((np.sign(position) == np.sign(response_choice))[trial_correct.astype(np.bool)])

    ibl_changepoint_df = dict(
        trial_num=trial_num,
        session_num=session_num,
        stim_probability_left=stim_probability_left,
        contrast=contrast,
        position=position,
        response_choice=response_choice,
        trial_correct=trial_correct,
        reaction_time=reaction_time)
    ibl_changepoint_df = pd.DataFrame(ibl_changepoint_df)
    ibl_changepoint_df.to_csv(
        os.path.join(log_dir, 'MIT_001.csv'),
        index=False)


def create_logger(log_dir):
    logging.basicConfig(
        filename=os.path.join(log_dir, 'logging.txt'),
        level=logging.DEBUG)
    # disable matplotlib font warnings
    logging.getLogger('matplotlib.font_manager').disabled = True


def create_model(model_str=None,
                 model_kwargs=None):
    # defaults
    if model_str is None:
        model_str = 'rnn'
    if model_kwargs is None:
        model_kwargs = dict(
            input_size=3,
            output_size=2,
            core_kwargs=dict(
                num_layers=1,
                hidden_size=50),
            param_init='default',
            connectivity_kwargs=dict(
                input_mask='none',
                recurrent_mask='none',
                readout_mask='none',
            ))

    if model_str in {'rnn', 'lstm', 'gru'}:
        model = RecurrentModel(
            model_str=model_str,
            model_kwargs=model_kwargs)
    elif model_str in {'ff'}:
        raise NotImplementedError
    else:
        raise NotImplementedError(f'Unknown core_str: {model_str}')

    return model


def create_optimizer(model,
                     optimizer_str='sgd',
                     optimizer_kwargs=None):
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(lr=0.0001)

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
        **optimizer_kwargs)

    return optimizer


def extract_session_data(envs):
    # calculate N back errors for each session
    N = 3
    for env in envs:
        env_session_data = env.session_data
        trial_end_data = env_session_data[env_session_data.trial_end == 1]
        for n in range(1, N + 1):
            # TODO: this doesn't work with multiple sessions
            col_name = f'{n}_back_correct'
            env_session_data[col_name] = np.nan
            n_back_correct_action_taken = trial_end_data.shift(
                periods=n).correct_action_taken
            env_session_data.loc[n_back_correct_action_taken.index.values, col_name] = \
                n_back_correct_action_taken

    # combine each environment's session data
    session_data = pd.concat([env.session_data for env in envs])

    # drop old repeated indices
    session_data.reset_index(inplace=True, drop=True)

    # calculate signed stimulus strength
    session_data['signed_trial_strength'] = session_data['trial_strength'] * \
                                            session_data['trial_side']

    session_data['concordant_trial'] = session_data['trial_side'] == \
                                       session_data['block_side']

    # record whether dt was in correct trial
    session_data['correct_trial_dt'] = np.nan
    for _, trial in session_data.groupby(['session_index', 'block_index', 'trial_index']):
        action_correct = trial.correct_action_taken.values[-1]
        trial_indices = trial.index
        session_data.loc[trial_indices, 'correct_trial_dt'] = action_correct

    # make trial side, block side orthogonal
    block_sides = session_data.block_side.values
    trial_sides = session_data.trial_side.values
    proj_trial_sides_onto_block_sides = np.dot(block_sides, trial_sides) * block_sides \
                                        / np.dot(block_sides, block_sides)
    trial_side_orthogonal = trial_sides - proj_trial_sides_onto_block_sides
    session_data['trial_side_orthogonal'] = trial_side_orthogonal

    # write CSV to disk for manual inspection, if curious
    # session_data.to_csv('session_data.csv', index=False)

    return session_data


def load_checkpoint(train_log_dir,
                    tensorboard_writer):
    # collect last checkpoint in the log directory
    checkpoint_paths = [os.path.join(train_log_dir, file_path)
                        for file_path in os.listdir(train_log_dir)
                        if file_path.endswith('.pt')]

    # select latest checkpoint path
    checkpoint_path = sorted(checkpoint_paths, key=os.path.getmtime)[-1]

    logging.info(f'Loading checkpoint at {checkpoint_path}')

    save_dict = torch.load(checkpoint_path)

    model = create_model(model_str=save_dict['model_str'],
                         model_kwargs=save_dict['model_kwargs'])
    model.load_state_dict(save_dict['model_state_dict'])

    optimizer = create_optimizer(
        model=model,
        optimizer_str='sgd')
    optimizer.load_state_dict(save_dict['optimizer_state_dict'])

    global_step = save_dict['global_step']

    # load environment kwargs
    with open(os.path.join(train_log_dir, 'notes.json')) as notes_fp:
        notes = json.load(notes_fp)
    env_kwargs = notes['env']

    # remove batch size - will use single session with large number of blocks
    del env_kwargs['batch_size']

    # replace some defaults
    # env_kwargs['trials_per_block_param'] = 1 / 65  # make longer blocks more common
    env_kwargs['blocks_per_session'] = 700
    # env_kwargs['blocks_per_session'] = 600
    # env_kwargs['blocks_per_session'] = 100
    # env_kwargs['blocks_per_session'] = 40

    return model, optimizer, global_step, env_kwargs


def run_envs(model,
             envs):
    total_reward = torch.zeros(1, dtype=torch.double, requires_grad=True)
    total_loss = torch.zeros(1, dtype=torch.double, requires_grad=True)
    if hasattr(model, 'reset_core_hidden'):
        model.reset_core_hidden()  # reset core's hidden state
    step_output = envs.reset()

    # step until any env finishes
    while not np.any(step_output['done']):
        total_reward = total_reward + torch.sum(step_output['reward'])  # cannot use +=
        total_loss = total_loss + torch.sum(step_output['loss'])

        model_output = model(step_output)

        # squeeze to remove the timestep (i.e. middle dimension) for the environment
        step_output = envs.step(
            actions=model_output['prob_output'],
            core_hidden=model_output['core_hidden'],
            model=model)

    envs.close()

    # can use any environment because they all step synchronously
    total_rnn_steps = envs[0].current_rnn_step_within_session

    # combine each environment's session data
    session_data = extract_session_data(envs=envs)

    avg_loss_per_dt = total_loss / (total_rnn_steps * len(envs))
    action_taken_by_total_trials = session_data[
        session_data.trial_end == 1.].action_taken.mean()
    correct_action_taken_by_action_taken = session_data[
        session_data.action_taken == 1.].correct_action_taken.mean()
    correct_action_taken_by_total_trials = session_data[
        session_data.trial_end == 1.].correct_action_taken.mean()
    feedback_by_dt = session_data.reward.mean()
    dts_by_trial = session_data.groupby([
        'session_index', 'block_index', 'trial_index']).size().mean()

    logging.info(f'Fraction of Correct Actions Taken by Total Actions Taken: '
                 f'{correct_action_taken_by_action_taken}')

    run_envs_output = dict(
        session_data=session_data,
        feedback_by_dt=feedback_by_dt,
        avg_loss_per_dt=avg_loss_per_dt,
        dts_by_trial=dts_by_trial,
        action_taken_by_total_trials=action_taken_by_total_trials,
        correct_action_taken_by_action_taken=correct_action_taken_by_action_taken,
        correct_action_taken_by_total_trials=correct_action_taken_by_total_trials
    )

    return run_envs_output


def save_train_output(test_or_train_output):
    raise NotImplementedError


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    logging.info(f'Seed: {seed}')


def stitch_plots(log_dir):

    plot_paths = [os.path.join(log_dir, file_name) for file_name in sorted(os.listdir(log_dir))
                  if file_name.endswith('.jpg') or file_name.endswith('.png')]

    images = [Image.open(plot_path) for plot_path in plot_paths]

    stitched_plots_path = os.path.join(
        log_dir,
        log_dir.split('/')[1] + '.pdf')

    images[0].save(
        stitched_plots_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=images[1:])

    # remove plots
    # for plot_path in plot_paths:
    #     os.remove(plot_path)

