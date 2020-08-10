import json
import logging
import numpy as np
import os
import torch

import utils.analysis
import utils.plot


def create_hook_fns_dict(hook_fns_frequencies,
                         start_grad_step,
                         num_grad_steps):

    # hook_fns_frequencies: list of (how many gradient steps per function call, function to call).
    # function must accept single input argument, hook_input.
    # Two unique values:
    #   0: run at start
    #   -1: run at end
    hooks_fn_dict = {}
    for freq, hook_fn in hook_fns_frequencies:

        # decide which step(s) to call hook at
        if freq == 0:
            hook_call_at_grad_steps = [start_grad_step]
        elif freq == -1:
            hook_call_at_grad_steps = [start_grad_step + num_grad_steps - 1]
        else:
            hook_call_at_grad_steps = np.arange(
                start=start_grad_step,
                stop=start_grad_step + num_grad_steps,
                step=freq,
                dtype=np.int)

        # add hook object reference to hooks_fn_dict at appropriate steps
        for grad_step in hook_call_at_grad_steps:
            if grad_step not in hooks_fn_dict:
                hooks_fn_dict[grad_step] = []
            hooks_fn_dict[grad_step].append(hook_fn)

    return hooks_fn_dict


def create_hook_fns_analyze(checkpoint_grad_step):
    hook_fns_frequencies = [
        (0, hook_write_scalars),
        (0, utils.plot.hook_plot_task_block_inference_multiple_blocks),
        (0, utils.plot.hook_plot_task_block_inference_single_block),
        # (0, utils.plot.hook_plot_bayesian_coupled_observer_state_space_trajectories_within_block),
        # (0, utils.plot.hook_plot_analysis_psytrack_fit),
        (0, utils.plot.hook_plot_behav_dts_per_trial_by_strength),
        (0, utils.plot.hook_plot_behav_bayesian_dts_per_trial_by_strength_correct_concordant),
        (0, utils.plot.hook_plot_behav_prob_correct_action_by_dts_within_trial),
        (0, utils.plot.hook_plot_behav_prob_correct_action_by_trial_within_block),
        (0, utils.plot.hook_plot_behav_prob_correct_action_by_zero_contrast_trial_within_block),
        (0, utils.plot.hook_plot_behav_prob_correct_by_strength_concordant),
        (0, utils.plot.hook_plot_behav_prob_correct_slope_intercept_by_prev_block_duration),
        (0, utils.plot.hook_plot_behav_reward_rate),
        (0, utils.plot.hook_plot_behav_right_action_by_signed_contrast),
        (0, utils.plot.hook_plot_behav_rnn_dts_per_trial_by_strength_correct_concordant),
        # (0, utils.plot.hook_plot_behav_right_action_after_error_by_right_action_after_correct),
        # (0, utils.plot.hook_plot_behav_subj_prob_block_switch_by_signed_trial_strength),
        (0, utils.plot.hook_plot_behav_trial_outcome_by_trial_strength),
        (0, utils.plot.hook_plot_compare_all_rnns_prob_correct_by_strength_concordant),
        (0, utils.plot.hook_plot_compare_all_rnns_prob_correct_by_trial_within_block),
        # (0, utils.plot.hook_plot_mice_reaction_time_by_strength_correct_concordant),
        # (0, utils.plot.hook_plot_mice_prob_correct_by_strength_trial_block),
        (0, utils.plot.hook_plot_model_effective_circuit),
        (0, utils.plot.hook_plot_model_hidden_unit_fraction_var_explained),
        (0, utils.plot.hook_plot_radd_behav_prob_correct_by_strength_concordant),
        (0, utils.plot.hook_plot_radd_behav_prob_correct_by_trial_within_block),
        (0, utils.plot.hook_plot_radd_state_space_distance_decoherence),
        (0, utils.plot.hook_plot_radd_state_space_trajectories_within_block),
        (0, utils.plot.hook_plot_radd_state_space_trajectories_within_trial),
        (0, utils.plot.hook_plot_state_space_effect_of_obs_along_task_aligned_vectors),
        (0, utils.plot.hook_plot_state_space_effect_of_feedback_along_task_aligned_vectors),
        (0, utils.plot.hook_plot_state_space_effect_of_feedback_along_task_aligned_vectors_by_side),
        (0, utils.plot.hook_plot_state_space_fixed_point_basins_of_attraction),
        (0, utils.plot.hook_plot_state_space_fixed_point_search),
        (0, utils.plot.hook_plot_state_space_projection_on_right_block_vector_by_trial_within_block),
        (0, utils.plot.hook_plot_state_space_projection_on_right_trial_vector_by_dts_within_trial),
        (0, utils.plot.hook_plot_state_space_trajectories_within_block),
        (0, utils.plot.hook_plot_state_space_trajectories_within_trial),
        (0, utils.plot.hook_plot_state_space_trials_by_classifier),
        (0, utils.plot.hook_plot_state_space_vector_fields_ideal),
        (0, utils.plot.hook_plot_state_space_vector_fields_real),
        # (0, utils.plot.hook_plot_pca_hidden_state_trajectories_controlled),
        # (0, utils.plot.hook_plot_hidden_to_hidden_jacobian_eigenvalues_complex_plane),
        # (0, utils.plot.hook_plot_hidden_to_hidden_jacobian_time_constants),
        (0, utils.plot.hook_plot_task_block_side_trial_side_by_trial_number),
        (0, utils.plot.hook_plot_task_stimuli_by_block_side),
        (0, utils.plot.hook_plot_task_stimuli_by_correct_trial_side),
        (0, utils.plot.hook_plot_task_stimuli_and_model_prob_in_first_n_trials),
    ]

    # every frequency must be zero
    for hook_fns_frequency, _ in hook_fns_frequencies:
        assert hook_fns_frequency == 0

    analyze_hooks = create_hook_fns_dict(
        hook_fns_frequencies=hook_fns_frequencies,
        start_grad_step=checkpoint_grad_step,
        num_grad_steps=0)

    return analyze_hooks


def create_hook_fns_train(start_grad_step,
                          num_grad_steps):

    plot_freq = 250

    hook_fns_frequencies = [
        (0, hook_log_params),
        (plot_freq, hook_print_model_progress),
        (100, hook_write_scalars),
        (0, utils.plot.hook_plot_task_block_side_trial_side_by_trial_number),
        # (plot_freq, utils.plot.hook_plot_task_block_side_trial_side_by_trial_number),
        # (plot_freq, utils.plot.hook_plot_task_stimuli_by_block_side),
        # (plot_freq, utils.plot.hook_plot_task_stimuli_by_correct_trial_side),
        # (plot_freq, utils.plot.hook_plot_task_stimuli_and_model_prob_in_first_n_trials),
        # (plot_freq, utils.plot.hook_plot_behav_dts_per_trial_by_stimuli_strength),
        # (plot_freq, utils.plot.hook_plot_behav_prob_correct_action_by_dts_within_trial),
        # (plot_freq, utils.plot.hook_plot_behav_prob_correct_action_by_trial_within_block),
        # (plot_freq, utils.plot.hook_plot_behav_prob_correct_action_on_block_side_trial_side_by_trial_strength),
        # (plot_freq, utils.plot.hook_plot_behav_prob_correct_slope_intercept_by_prev_block_duration),
        # (plot_freq, utils.plot.hook_plot_behav_right_action_by_signed_contrast),
        # (plot_freq, utils.plot.hook_plot_behav_subj_prob_block_switch_by_signed_trial_strength),
        # (plot_freq, utils.plot.hook_plot_behav_trial_outcome_by_trial_strength),
        # (plot_freq, utils.plot.hook_plot_model_effective_circuit),
        # (plot_freq, utils.plot.hook_plot_model_hidden_unit_fraction_var_explained),
        # (plot_freq, utils.plot.hook_plot_model_weights_and_gradients),
        # (plot_freq, utils.plot.hook_plot_model_weights_community_detection),
        # (10, utils.plot.hook_plot_pca_hidden_state_activity_within_block),
        # (plot_freq, utils.plot.hook_plot_pca_hidden_state_vector_fields),
        # (plot_freq, utils.plot.hook_plot_state_space_trajectories_within_trial),
        # (plot_freq, utils.plot.hook_plot_state_space_trajectories_within_block),
        # (10, utils.plot.hook_plot_hidden_state_projected_trajectories_controlled),
        # (plot_freq, utils.plot.hook_plot_pca_hidden_state_fixed_points),
        # (10, utils.plot.hook_plot_psytrack_fit),
        # (10, utils.plot.hook_plot_hidden_to_hidden_jacobian_eigenvalues_complex_plane),
        (5000, hook_save_model),
    ]

    train_hooks = create_hook_fns_dict(
        hook_fns_frequencies=hook_fns_frequencies,
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    return train_hooks


def hook_log_params(hook_input):

    env_dict = dict(
        batch_size=len(hook_input['envs']),
        block_side_probs=hook_input['envs'][0].block_side_probs,
        trials_per_block_param=hook_input['envs'][0].trials_per_block_param,
        possible_trial_strengths=hook_input['envs'][0].possible_trial_strengths,
        possible_trial_strengths_probs=hook_input['envs'][0].possible_trial_strengths_probs,
        blocks_per_session=hook_input['envs'][0].blocks_per_session,
        min_trials_per_block=hook_input['envs'][0].min_trials_per_block,
        max_trials_per_block=hook_input['envs'][0].max_trials_per_block,
        max_stimuli_per_trial=hook_input['envs'][0].max_obs_per_trial,
        rnn_steps_before_stimulus=hook_input['envs'][0].rnn_steps_before_obs,
        time_delay_penalty=hook_input['envs'][0].time_delay_penalty,
    )

    notes_file = os.path.join(
        hook_input['tensorboard_writer'].get_logdir(),
        'params.json')
    with open(notes_file, "w") as f:
        notes_str = json.dumps(hook_input['params'], indent=4, sort_keys=True)
        print(notes_str)
        f.write(notes_str)


def hook_print_model_progress(hook_input):
    print('Grad Step: {:5d}\n'
          'Loss per dt: {:6.3f}\n'
          'Correct Action/Action Taken: {:6.3f}\tdts per Trial: {:6.3f}'.format(
        hook_input['grad_step'],
        hook_input['avg_loss_per_dt'],
        hook_input['correct_action_taken_by_action_taken'],
        hook_input['dts_by_trial']))


def hook_save_model(hook_input):
    model = hook_input['model']
    grad_step = hook_input['grad_step']

    save_dict = dict(
        model_str=model.model_str,
        model_kwargs=model.model_kwargs,
        input_size=model.input_size,
        output_size=model.output_size,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=hook_input['optimizer'].state_dict(),
        global_step=grad_step)

    checkpoint_file_path = os.path.join(
        hook_input['tensorboard_writer'].get_logdir(),
        'checkpoint_grad_steps={:04d}.pt'.format(
            hook_input['grad_step']))

    torch.save(
        obj=save_dict,
        f=checkpoint_file_path)

    print('Saved model!')


def hook_write_scalars(hook_input):

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'feedback_by_dt',
        scalar_value=hook_input['feedback_by_dt'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'avg_loss_per_dt',
        scalar_value=hook_input['avg_loss_per_dt'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'dts_by_trial',
        scalar_value=hook_input['dts_by_trial'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'action_taken_by_total_trials',
        scalar_value=hook_input['action_taken_by_total_trials'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'correct_action_taken_by_action_taken',
        scalar_value=hook_input['correct_action_taken_by_action_taken'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'correct_action_taken_by_total_trials',
        scalar_value=hook_input['correct_action_taken_by_total_trials'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'avg_trial_side',
        scalar_value=hook_input['session_data'].trial_side.mean(),
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'avg_trial_side_left_block',
        scalar_value=hook_input['session_data'][
            hook_input['session_data'].block_side == -1].trial_side.mean(),
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'avg_trial_side_right_block',
        scalar_value=hook_input['session_data'][
            hook_input['session_data'].block_side == 1].trial_side.mean(),
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'min_trial_per_block',
        scalar_value=min([min(hook_input['envs'][i].num_trials_per_block)
                          for i in range(len(hook_input['envs']))]),
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'max_trial_per_block',
        scalar_value=max([max(hook_input['envs'][i].num_trials_per_block)
                          for i in range(len(hook_input['envs']))]),
        global_step=hook_input['grad_step'])

    # plot the variance, fraction of variance for the first 4 PCs (arbitrary cutoff)
    num_pcs_to_plot = min(4, len(hook_input['variance_explained']))
    total_variance = np.sum(hook_input['variance_explained'])
    for i in range(num_pcs_to_plot):
        hook_input['tensorboard_writer'].add_scalar(
            tag=hook_input['tag_prefix'] + f'pc/frac_variance_explained_{i + 1}',
            scalar_value=hook_input['variance_explained'][i] / total_variance,
            global_step=hook_input['grad_step'])

        hook_input['tensorboard_writer'].add_scalar(
            tag=hook_input['tag_prefix'] + f'pc/variance_explained_{i + 1}',
            scalar_value=hook_input['variance_explained'][i],
            global_step=hook_input['grad_step'])


def hook_write_parameter_histograms(hook_input):
    # record parameter distributions
    for param_name, param in hook_input['model'].named_parameters():
        if param.requires_grad and param.grad is not None:
            hook_input['tensorboard_writer'].add_histogram(
                tag=hook_input['tag_prefix'] + param_name,
                values=param.data,
                global_step=hook_input['grad_step'])


def hook_write_pr_curve(hook_input):
    hook_input['tensorboard_writer'].add_pr_curve(
        tag=hook_input['tag_prefix'] + 'pr_curve_per_grad_step',
        labels=(1 + hook_input['run_envs_output']['trial_data']['stimuli_sides'].to_numpy()) // 2,
        predictions=hook_input['run_envs_output']['trial_data']['model_correct_action_probs'].to_numpy(),
        global_step=hook_input['grad_step'])
