import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torch.utils.tensorboard import SummaryWriter

from utils.analysis import add_analysis_data_to_hook_input
from utils.env import create_biased_choice_worlds
from utils.hooks import create_hook_fns_analyze
from utils.run import load_checkpoint, run_envs, set_seed, stitch_plots


def main():
    seed = 1
    set_seed(seed=seed)

    run_dir = 'rnn, block_side_probs=0.50, snr=0.9'
    train_log_dir = os.path.join('runs', run_dir)
    analyze_log_dir = os.path.join('runs', 'analyze_' + run_dir)
    tensorboard_writer = SummaryWriter(log_dir=analyze_log_dir)

    # set stdout to specified text file
    sys.stdout = open(os.path.join(analyze_log_dir, 'stdout.txt'), 'w')

    model, optimizer, grad_step, env_kwargs = load_checkpoint(
        train_log_dir=train_log_dir,
        tensorboard_writer=tensorboard_writer)

    envs = create_biased_choice_worlds(
        num_sessions=1,
        **env_kwargs)

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

    # w_recurrent = model.core.weight_hh_l0.detach().numpy()
    # w_recurrent_svd_results = np.linalg.svd(w_recurrent)
    # w_in = model.core.weight_ih_l0.detach().numpy()
    # w_in_svd_results = np.linalg.svd(w_in)

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
        # fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli,
        tensorboard_writer=tensorboard_writer,
        tag_prefix=tag_prefix,
        seed=seed)

    add_analysis_data_to_hook_input(hook_input=hook_input)

    for hook_fn in hook_fns[start_grad_step]:
        hook_fn(hook_input)

        # save figures to disk
        fn_name = str(hook_fn).split(' ')[1] + '.jpg'
        fig = plt.gcf()  # load whatever figure was created by hook_fn
        fig.savefig(os.path.join(tensorboard_writer.log_dir, fn_name))
        plt.close(fig)

    # stitch figures together into single PDF for easy use
    stitch_plots(log_dir=tensorboard_writer.log_dir)

    return analyze_model_output


if __name__ == '__main__':
    main()

