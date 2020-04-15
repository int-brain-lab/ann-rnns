import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from utils.analysis import compute_model_fixed_points, compute_hidden_states_pca, \
    compute_eigenvalues_svd
from utils.env import create_biased_choice_worlds
from utils.hooks import create_hook_fns_analyze
from utils.run import load_checkpoint, run_envs, set_seed


def main():
    seed = 1
    set_seed(seed=seed)

    run_dir = 'rnn, num_layers=1, hidden_size=50, param_init=default, input_mask=none, recurrent_mask=none, readout_mask=none_2020-04-09 01:13:16.774657'
    train_log_dir = os.path.join('runs', run_dir)
    analyze_log_dir = os.path.join('runs', 'analyze_' + run_dir)
    tensorboard_writer = SummaryWriter(log_dir=analyze_log_dir)

    envs = create_biased_choice_worlds(
        num_sessions=35,
        blocks_per_session=10)

    model, optimizer, grad_step = load_checkpoint(
        train_log_dir=train_log_dir,
        tensorboard_writer=tensorboard_writer)

    hook_fns = create_hook_fns_analyze(
        start_grad_step=grad_step)

    analyze_model_output = analyze_model(
        model=model,
        envs=envs,
        optimizer=optimizer,
        hook_fns=hook_fns,
        seed=seed,
        tensorboard_writer=tensorboard_writer,
        start_grad_step=grad_step,
        num_grad_steps=0,
        tag_prefix='analyze/')

    tensorboard_writer.close()


def analyze_model(model,
                  envs,
                  optimizer,
                  hook_fns,
                  seed,
                  tensorboard_writer,
                  start_grad_step,
                  num_grad_steps=0,
                  tag_prefix='analyze/'):

    if num_grad_steps != 0:
        raise ValueError('Number of gradient steps must be zero!')

    run_envs_output = run_envs(
        model=model,
        envs=envs)

    analyze_model_output = dict(
        global_step=start_grad_step,
        run_envs_output=run_envs_output
    )

    hidden_states = np.stack(
        [hidden_state for hidden_state in
         run_envs_output['session_data']['hidden_state'].values])

    pca_hidden_states, pca_readout_weights, pca_xrange, pca_yrange, pca = \
        compute_hidden_states_pca(
            hidden_states=hidden_states.reshape(hidden_states.shape[0], -1),
            readout_weights=model.readout.weight.data.numpy())

    variance_explained, frac_variance_explained = compute_eigenvalues_svd(
        matrix=hidden_states.reshape(hidden_states.shape[0], -1))

    # fixed_points_by_side_by_stimuli = compute_model_fixed_points(
    #     model=model,
    #     pca=pca,
    #     pca_hidden_states=pca_hidden_states,
    #     session_data=run_envs_output['session_data'],
    #     hidden_states=hidden_states,
    #     num_grad_steps=50)

    hook_input = dict(
        feedback_by_dt=run_envs_output['feedback_by_dt'],
        avg_loss_per_dt=run_envs_output['avg_loss_per_dt'].item(),
        dts_by_trial=run_envs_output['dts_by_trial'],
        action_taken_by_total_trials=run_envs_output['action_taken_by_total_trials'],
        correct_action_taken_by_action_taken=run_envs_output['correct_action_taken_by_action_taken'],
        session_data=run_envs_output['session_data'],
        hidden_states=hidden_states,
        grad_step=start_grad_step,
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
        # fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli,
        tensorboard_writer=tensorboard_writer,
        tag_prefix=tag_prefix,
        seed=seed)

    for hook_fn in hook_fns[start_grad_step]:
        hook_fn(hook_input)

    return analyze_model_output


if __name__ == '__main__':
    main()
