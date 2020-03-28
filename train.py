from datetime import datetime
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.analysis import compute_model_fixed_points, compute_hidden_states_pca, \
    compute_eigenvalues_svd
from utils.env import create_biased_choice_worlds
from utils.hooks import create_hook_fns_train
from utils.run import create_model, create_optimizer, run_envs, set_seed


def main():
    seed = 1
    set_seed(seed=seed)

    model = create_model()

    log_dir = os.path.join('runs', model.description_str + '_' + str(datetime.now()))
    tensorboard_writer = SummaryWriter(log_dir=log_dir)

    optimizer = create_optimizer(
        model=model,
        optimizer_str='sgd')

    envs = create_biased_choice_worlds(num_sessions=1)

    start_grad_step = 0
    num_grad_steps = 25001

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
        run_envs_output['avg_loss'].backward()
        optimizer.step()

        if grad_step in hook_fns:

            hidden_states = np.stack(
                [hidden_state for hidden_state in
                 run_envs_output['session_data']['hidden_state'].values])

            pca_hidden_states, pca_readout_weights, pca_xrange, pca_yrange, pca = \
                compute_hidden_states_pca(
                    hidden_states=hidden_states.reshape(hidden_states.shape[0], -1),
                    readout_weights=model.readout.weight.data.numpy())

            variance_explained, frac_variance_explained = compute_eigenvalues_svd(
                matrix=hidden_states.reshape(hidden_states.shape[0], -1))

            fixed_points_by_side_by_stimuli = compute_model_fixed_points(
                model=model,
                pca=pca,
                pca_hidden_states=pca_hidden_states,
                session_data=run_envs_output['session_data'],
                hidden_states=hidden_states,
                num_grad_steps=50)

            hook_input = dict(
                avg_correct_action_prob=run_envs_output['avg_correct_action_prob'],
                avg_loss=run_envs_output['avg_loss'].item(),
                avg_reward=run_envs_output['avg_reward'].item(),
                avg_rnn_steps_per_trial=run_envs_output['avg_rnn_steps_per_trial'],
                session_data=run_envs_output['session_data'],
                hidden_states=hidden_states,
                grad_step=grad_step,
                model=model,
                envs=envs,
                optimizer=optimizer,
                variance_explained=variance_explained,
                frac_variance_explained=frac_variance_explained,
                pca_hidden_states=pca_hidden_states,
                pca_readout_weights=pca_readout_weights,
                pca_xrange=pca_xrange,
                pca_yrange=pca_yrange,
                pca=pca,
                fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli,
                tensorboard_writer=tensorboard_writer,
                tag_prefix=tag_prefix,
                seed=seed)

            for hook_fn in hook_fns[grad_step]:
                # try:
                hook_fn(hook_input)
                # except Exception as e:
                #     print(e)

    train_model_output = dict(
        grad_step=grad_step,
        run_envs_output=run_envs_output
    )

    return train_model_output


if __name__ == '__main__':
    log_dir = 'runs'
    os.makedirs(log_dir, exist_ok=True)
    main()
