from datetime import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

import utils.env
import utils.hooks
import utils.models
import utils.params


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


def create_logger(run_dir):

    logging.basicConfig(
        filename=os.path.join(run_dir, 'logging.log'),
        level=logging.DEBUG)

    logging.info('Logger created successfully')

    # also log to std out
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)

    # disable matplotlib font warnings
    # logging.getLogger('matplotlib.font_manager').disabled = True


def create_loss_fn(loss_fn_params):

    if loss_fn_params['loss_fn'] == 'mse':
        return torch.nn.MSELoss()
    elif loss_fn_params['loss_fn'] == 'ce':
        return torch.nn.CrossEntropyLoss()
    elif loss_fn_params['loss_fn'] == 'nll':
        return torch.nn.NLLLoss()
    else:
        raise NotImplementedError


def create_model(model_params):
    if model_params['architecture'] in {'rnn', 'lstm', 'gru'}:
        model = utils.models.RecurrentModel(
            model_architecture=model_params['architecture'],
            model_kwargs=model_params['kwargs'])
    else:
        raise NotImplementedError
    return model


def create_optimizer(model,
                     optimizer_params):

    if optimizer_params['optimizer'] == 'sgd':
        optimizer_constructor = torch.optim.SGD
    elif optimizer_params['optimizer'] == 'adam':
        optimizer_constructor = torch.optim.Adam
    elif optimizer_params['optimizer'] == 'rmsprop':
        optimizer_constructor = torch.optim.RMSprop
    else:
        raise NotImplementedError('Unknown optimizer string')

    optimizer = optimizer_constructor(
        params=model.parameters(),
        **optimizer_params['kwargs'])

    return optimizer


def create_params_analyze(train_run_dir):

    # load parameters
    with open(os.path.join(train_run_dir, 'params.json')) as params_fp:
        params = json.load(params_fp)

    # remove batch size - will use single session with large number of blocks
    params['env']['num_sessions'] = 1

    # replace some defaults
    # env_kwargs['trials_per_block_param'] = 1 / 65  # make longer blocks more common
    params['env']['kwargs']['blocks_per_session'] = 1000
    # params['env']['kwargs']['blocks_per_session'] = 100
    # params['env']['kwargs']['blocks_per_session'] = 50

    return params


def create_params_train():
    params = utils.params.train_params
    return params


def create_run_id(params):

    included_params = [
        params['model']['architecture'],
        'max_stim_strength=' + str(params['env']['kwargs']['max_stimulus_strength']),
        'hidden_size=' + str(params['model']['kwargs']['core_kwargs']['hidden_size']),
    ]
    separator = ', '
    run_id = separator.join(str(ip) for ip in included_params)
    return run_id


def create_tensorboard_writer(run_dir):
    tensorboard_writer = SummaryWriter(
        log_dir=run_dir)
    return tensorboard_writer


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


def load_checkpoint(train_run_dir,
                    params):
    # collect last checkpoint in the log directory
    checkpoint_paths = [os.path.join(train_run_dir, file_path)
                        for file_path in os.listdir(train_run_dir)
                        if file_path.endswith('.pt')]

    # select latest checkpoint path
    checkpoint_path = sorted(checkpoint_paths, key=os.path.getmtime)[-1]

    logging.info(f'Loading checkpoint at {checkpoint_path}')

    save_dict = torch.load(checkpoint_path)

    model = create_model(model_params=params['model'])
    model.load_state_dict(save_dict['model_state_dict'])

    optimizer = create_optimizer(
        model=model,
        optimizer_params=params['optimizer'])
    optimizer.load_state_dict(save_dict['optimizer_state_dict'])

    global_step = save_dict['global_step']

    return model, optimizer, global_step


def run_envs(model,
             envs,
             log_results=True):

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
            actions_logits=model_output['linear_output'],
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

    correct_action_taken_by_total_steps = np.mean(
        session_data.correct_action_taken.fillna(0).values)

    feedback_by_dt = session_data.reward.mean()

    dts_by_trial = session_data.groupby([
        'session_index', 'block_index', 'trial_index']).size().mean()

    if log_results:

        logging.info(f'Average Loss Per Dt: {avg_loss_per_dt.item()}')

        logging.info(f'# Action Trials / # Total Trials: '
                     f'{action_taken_by_total_trials}')

        logging.info(f'# Correct Trials / # Action Trials: '
                     f'{correct_action_taken_by_action_taken}')

        logging.info(f'# Correct Trials / # Total Trials: '
                     f'{correct_action_taken_by_total_trials}')

        logging.info(f'# Correct Trials / # Total Steps: '
                     f'{correct_action_taken_by_total_steps}')

        logging.info(f'Average steps per trial: '
                     f'{dts_by_trial}')

    run_envs_output = dict(
        session_data=session_data,
        feedback_by_dt=feedback_by_dt,
        avg_loss_per_dt=avg_loss_per_dt,
        dts_by_trial=dts_by_trial,
        action_taken_by_total_trials=action_taken_by_total_trials,
        correct_action_taken_by_action_taken=correct_action_taken_by_action_taken,
        correct_action_taken_by_total_trials=correct_action_taken_by_total_trials,
        correct_action_taken_by_total_steps=correct_action_taken_by_total_steps,
    )

    return run_envs_output


def save_train_output(test_or_train_output):
    raise NotImplementedError


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    logging.info(f'Seed: {seed}')


def setup_analyze(train_run_id):

    run_dir = 'runs'
    train_run_dir = os.path.join(run_dir, train_run_id)
    analyze_run_dir = os.path.join(train_run_dir, 'analyze')
    os.makedirs(analyze_run_dir, exist_ok=True)
    create_logger(run_dir=analyze_run_dir)
    params = create_params_analyze(train_run_dir=train_run_dir)
    set_seeds(seed=params['run']['seed'])
    tensorboard_writer = create_tensorboard_writer(
        run_dir=analyze_run_dir)
    model, optimizer, checkpoint_grad_step = load_checkpoint(
        train_run_dir=train_run_dir,
        params=params)
    loss_fn = create_loss_fn(
        loss_fn_params=params['loss_fn'])
    fn_hook_dict = utils.hooks.create_hook_fns_analyze(
        checkpoint_grad_step=checkpoint_grad_step)
    envs = utils.env.create_biased_choice_worlds(
        env_params=params['env'],
        base_loss_fn=loss_fn)
    setup_results = dict(
        params=params,
        run_id=train_run_id,
        run_dir=run_dir,
        tensorboard_writer=tensorboard_writer,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        fn_hook_dict=fn_hook_dict,
        envs=envs,
        checkpoint_grad_step=checkpoint_grad_step,
    )
    return setup_results


def setup_train():

    log_dir = 'runs'
    os.makedirs(log_dir, exist_ok=True)

    params = create_params_train()
    run_id = create_run_id(params=params)
    run_dir = os.path.join(log_dir, run_id + '_' + str(datetime.now()))
    os.makedirs(run_dir, exist_ok=True)

    create_logger(run_dir=run_dir)
    set_seeds(seed=params['run']['seed'])
    tensorboard_writer = create_tensorboard_writer(
        run_dir=run_dir)
    model = create_model(
        model_params=params['model'])
    optimizer = create_optimizer(
        model=model,
        optimizer_params=params['optimizer'])
    loss_fn = create_loss_fn(
        loss_fn_params=params['loss_fn'])
    fn_hook_dict = utils.hooks.create_hook_fns_train(
        start_grad_step=params['run']['start_grad_step'],
        num_grad_steps=params['run']['num_grad_steps'])
    envs = utils.env.create_biased_choice_worlds(
        env_params=params['env'],
        base_loss_fn=loss_fn)

    setup_results = dict(
        params=params,
        run_id=run_id,
        run_dir=run_dir,
        tensorboard_writer=tensorboard_writer,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        fn_hook_dict=fn_hook_dict,
        envs=envs
    )
    return setup_results


def stitch_plots(log_dir):

    plot_paths = [os.path.join(log_dir, file_name) for file_name in sorted(os.listdir(log_dir))
                  if file_name.endswith('.jpg') or file_name.endswith('.png')]

    images = [Image.open(plot_path) for plot_path in plot_paths]

    stitched_plots_path = os.path.join(
        log_dir,
        'analyze_' + log_dir.split('/')[1] + '.pdf')

    images[0].save(
        stitched_plots_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=images[1:])

    # remove plots
    # for plot_path in plot_paths:
    #     os.remove(plot_path)

