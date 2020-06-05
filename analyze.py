import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from utils.analysis import add_analysis_data_to_hook_input
from utils.env import create_biased_choice_worlds
from utils.hooks import create_hook_fns_analyze
from utils.run import convert_session_data_to_ibl_changepoint_csv, create_logger, \
    load_checkpoint, run_envs, set_seed, stitch_plots


def main():

    run_dir = 'rnn, block_side_probs=0.80, snr=2.5'
    # run_dir = 'rnn, block_side_probs=0.80, snr=2.5, hidden_size=2'
    train_log_dir = os.path.join('runs', run_dir)
    analyze_log_dir = os.path.join('runs', 'analyze_' + run_dir)
    tensorboard_writer = SummaryWriter(log_dir=analyze_log_dir)
    create_logger(log_dir=analyze_log_dir)
    seed = 1
    set_seed(seed=seed)

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

    # stitch figures together into single PDF for easy use
    stitch_plots(log_dir=tensorboard_writer.log_dir)

    # convert_session_data_to_ibl_changepoint_csv(
    #     session_data=analyze_model_output['run_envs_output']['session_data'],
    #     env_block_side_probs=envs[0].block_side_probs,
    #     log_dir=analyze_log_dir)

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
        correct_action_taken_by_total_trials=run_envs_output['correct_action_taken_by_total_trials'],
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
        fig.savefig(os.path.join(tensorboard_writer.log_dir, fn_name),
                    bbox_inches='tight')  # removes surrounding whitespace
        plt.close(fig)

    # recurrent_jacobian = hook_input['fixed_point_df']['jacobian_hidden'][0]
    # import scipy.linalg
    # eigresult = scipy.linalg.eig(recurrent_jacobian, left=True, right=True)
    #
    # recurrent_matrix = hook_input['model'].core.weight_hh_l0.data.numpy()
    # recurrent_eigresult = scipy.linalg.eig(recurrent_matrix, left=True, right=True)
    # np.sort(np.real(eigresult[0]))[::-1]
    # input_matrix = hook_input['model'].core.weight_ih_l0.data.numpy()
    # input_eigresult = scipy.linalg.eig(recurrent_matrix, left=True, right=True)
    # np.sort(np.real(input_eigresult[0]))[::-1]

    return analyze_model_output


if __name__ == '__main__':
    main()

