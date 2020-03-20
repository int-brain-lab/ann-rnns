import numpy as np
import os
import pandas as pd
import torch

from utils.models import RecurrentModel


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

    # model = create_model(
    #     model_str='ff',
    #     model_kwargs=dict(ff_kwargs=dict(activation_str='relu',
    #                                      layer_widths=[10, 10])))

    if model_str in {'rnn', 'lstm', 'gru'}:
        model = RecurrentModel(
            model_str=model_str,
            model_kwargs=model_kwargs)
    elif model_str in {'ff'}:
        raise NotImplementedError
        # model = FeedforwardModel(
        #     model_str=model_str,
        #     model_kwargs=model_kwargs)
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


def load_checkpoint(train_log_dir,
                    tensorboard_writer):

    # collect last checkpoint in the log directory
    checkpoint_paths = [os.path.join(train_log_dir, file_path)
                        for file_path in os.listdir(train_log_dir)
                        if file_path.endswith('.pt')]

    # select latest checkpoint path
    checkpoint_path = sorted(checkpoint_paths, key=os.path.getmtime)[-1]

    print(f'Loading checkpoint at {checkpoint_path}')

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

    # step until any env finishes
    while not np.any(step_output['done']):

        total_reward = total_reward + torch.sum(step_output['reward'])  # cannot use +=
        total_loss = total_loss + torch.sum(step_output['loss'])

        model_output = model(step_output)

        # squeeze to remove the timestep (i.e. middle dimension) for the environment
        step_output = envs.step(
            actions=model_output['softmax_output'],
            core_hidden=model_output['core_hidden'],
            model=model)

    envs.close()

    # can use any environment because they all step synchronously
    total_rnn_steps = envs[0].current_rnn_step_within_session

    # combine each environment's session data
    session_data = pd.concat([env.session_data for env in envs])
    # check that no nan rows remain
    rows_containing_nan = session_data.isnull().any(axis=1)
    assert rows_containing_nan.sum() == 0

    # write CSV to disk for manual inspection, if curious
    # session_data.to_csv('session_data.csv', index=False)

    # divide by total trials, batch size
    avg_reward = total_reward / (total_rnn_steps * len(envs))
    avg_loss = total_loss / (total_rnn_steps * len(envs))

    avg_rnn_steps_per_trial = session_data.groupby([
        'session_index', 'block_index', 'trial_index']).size().mean()

    run_envs_output = dict(
        session_data=session_data,
        avg_reward=avg_reward,
        avg_loss=avg_loss,
        avg_rnn_steps_per_trial=avg_rnn_steps_per_trial
    )

    return run_envs_output


def save_train_output(test_or_train_output):
    raise NotImplementedError


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
