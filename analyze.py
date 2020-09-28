import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from utils.analysis import add_analysis_data_to_hook_input
from utils.plot import run_hook_and_save_fig
import utils.run


def analyze():

    # train_run_id = 'rnn, block_side_probs=0.80, snr=2.5'
    train_run_id = 'rnn, max_stim_strength=2.5, hidden_size=250_2020-08-11 14:05:02.536133'
    setup_results = utils.run.setup_analyze(
        train_run_id=train_run_id)

    analyze_model(
        model=setup_results['model'],
        envs=setup_results['envs'],
        optimizer=setup_results['optimizer'],
        fn_hook_dict=setup_results['fn_hook_dict'],
        params=setup_results['params'],
        tensorboard_writer=setup_results['tensorboard_writer'],
        checkpoint_grad_step=setup_results['checkpoint_grad_step'])

    # convert_session_data_to_ibl_changepoint_csv(
    #     session_data=analyze_model_output['run_envs_output']['session_data'],
    #     env_block_side_probs=envs[0].block_side_probs,
    #     log_dir=analyze_log_dir)

    setup_results['tensorboard_writer'].close()
    logging.info('Completed successfully')


def analyze_model(model,
                  envs,
                  optimizer,
                  fn_hook_dict,
                  params,
                  tensorboard_writer,
                  checkpoint_grad_step,
                  tag_prefix='analyze/'):

    logging.info('Running high dimensional, task-trained model...')
    run_envs_output = utils.run.run_envs(
        model=model,
        envs=envs)

    analyze_model_output = dict(
        global_step=checkpoint_grad_step,
        run_envs_output=run_envs_output
    )

    hidden_states = np.stack(
        [hidden_state for hidden_state in
         run_envs_output['session_data']['hidden_state'].values])

    hook_input = dict(
        feedback_by_dt=run_envs_output['feedback_by_dt'],
        avg_loss_per_dt=run_envs_output['avg_loss_per_dt'].item(),
        dts_by_trial=run_envs_output['dts_by_trial'],
        action_taken_by_total_trials=run_envs_output['action_taken_by_total_trials'],
        correct_action_taken_by_action_taken=run_envs_output['correct_action_taken_by_action_taken'],
        correct_action_taken_by_total_trials=run_envs_output['correct_action_taken_by_total_trials'],
        session_data=run_envs_output['session_data'],
        hidden_states=hidden_states,
        grad_step=checkpoint_grad_step,
        model=model,
        envs=envs,
        optimizer=optimizer,
        # fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli,
        tensorboard_writer=tensorboard_writer,
        params=params,
        tag_prefix=tag_prefix)

    add_analysis_data_to_hook_input(hook_input=hook_input)

    for hook_fn in fn_hook_dict[checkpoint_grad_step]:
        run_hook_and_save_fig(hook_fn=hook_fn, hook_input=hook_input)

    # stitch figures together into single PDF for easy use
    utils.run.stitch_plots(log_dir=tensorboard_writer.log_dir)

    return analyze_model_output


if __name__ == '__main__':
    analyze()
