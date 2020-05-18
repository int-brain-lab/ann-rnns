from datetime import datetime
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.analysis import compute_eigenvalues
from utils.env import create_biased_choice_worlds
from utils.hooks import create_hook_fns_train
from utils.run import create_logger, create_model, create_optimizer, run_envs, set_seed


def main():
    seed = 1
    set_seed(seed=seed)

    model = create_model()

    train_log_dir = os.path.join('runs', model.description_str + '_' + str(datetime.now()))
    tensorboard_writer = SummaryWriter(log_dir=train_log_dir)
    create_logger(log_dir=train_log_dir)

    optimizer = create_optimizer(
        model=model,
        optimizer_str='sgd',
        optimizer_kwargs=dict(lr=0.001,
                              momentum=0.1))

    envs = create_biased_choice_worlds(
        num_sessions=1)

    start_grad_step = 0
    num_grad_steps = 35001

    hook_fns = create_hook_fns_train(
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    train_model_output = train_model(
        model=model,
        envs=envs,
        optimizer=optimizer,
        hook_fns=hook_fns,
        seed=seed,
        tensorboard_writer=tensorboard_writer,
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    tensorboard_writer.close()


def train_model(model,
                envs,
                optimizer,
                hook_fns,
                seed,
                tensorboard_writer,
                start_grad_step=0,
                num_grad_steps=150,
                tag_prefix='train/'):

    # sets the model in training mode.
    model.train()

    # ensure assignment before reference
    run_envs_output = {}
    grad_step = start_grad_step

    for grad_step in range(start_grad_step, start_grad_step + num_grad_steps):
        if hasattr(model, 'apply_connectivity_masks'):
            model.apply_connectivity_masks()
        if hasattr(model, 'reset_core_hidden'):
            model.reset_core_hidden()
        optimizer.zero_grad()
        run_envs_output = run_envs(
            model=model,
            envs=envs)
        run_envs_output['avg_loss_per_dt'].backward()
        optimizer.step()

        if grad_step in hook_fns:

            hidden_states = np.stack(
                [hidden_state for hidden_state in
                 run_envs_output['session_data']['hidden_state'].values])

            hook_input = dict(
                feedback_by_dt=run_envs_output['feedback_by_dt'],
                avg_loss_per_dt=run_envs_output['avg_loss_per_dt'].item(),
                dts_by_trial=run_envs_output['dts_by_trial'],
                action_taken_by_total_trials=run_envs_output['action_taken_by_total_trials'],
                correct_action_taken_by_action_taken=run_envs_output['correct_action_taken_by_action_taken'],
                session_data=run_envs_output['session_data'],
                hidden_states=hidden_states,
                grad_step=grad_step,
                model=model,
                envs=envs,
                optimizer=optimizer,
                tensorboard_writer=tensorboard_writer,
                tag_prefix=tag_prefix,
                seed=seed)

            eigenvalues_svd_results = compute_eigenvalues(
                matrix=hook_input['hidden_states'].reshape(hook_input['hidden_states'].shape[0], -1))

            hook_input.update(eigenvalues_svd_results)

            for hook_fn in hook_fns[grad_step]:
                hook_fn(hook_input)

    train_model_output = dict(
        grad_step=grad_step,
        run_envs_output=run_envs_output
    )

    return train_model_output


if __name__ == '__main__':
    log_dir = 'runs'
    os.makedirs(log_dir, exist_ok=True)
    main()
