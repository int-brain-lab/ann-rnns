import json
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


def create_hook_fns_analyze(start_grad_step):

    hook_fns_frequencies = [
        # (0, utils.plot.hook_plot_model_community_detection),
        (0, hook_write_scalars),
        (0, utils.plot.hook_plot_within_trial_stimuli_and_model_prob),
        (0, utils.plot.hook_plot_fraction_var_explained),
        (0, utils.plot.hook_plot_avg_model_prob_by_trial_within_block),
        (0, utils.plot.hook_plot_psychometric_curves),
        # (0, utils.plot.hook_plot_hidden_state_correlations),
        (0, utils.plot.hook_plot_pca_hidden_state_fixed_points),
        # (0, utils.plot.hook_plot_pca_hidden_state_activity_within_block),
        (0, utils.plot.hook_plot_pca_hidden_state_vector_fields),
        (0, utils.plot.hook_plot_pca_hidden_state_trajectories_within_block),
        # (0, utils.plot.hook_plot_pca_hidden_state_trajectories_controlled),
        # (0, utils.plot.hook_plot_psytrack_fit),
        # (0, utils.plot.hook_plot_hidden_to_hidden_jacobian_eigenvalues_complex_plane),
        # (0, utils.plot.hook_plot_hidden_to_hidden_jacobian_time_constants),
    ]

    # every frequency must be zero
    for hook_fns_frequency, _ in hook_fns_frequencies:
        assert hook_fns_frequency == 0

    analyze_hooks = create_hook_fns_dict(
        hook_fns_frequencies=hook_fns_frequencies,
        start_grad_step=start_grad_step,
        num_grad_steps=0)

    return analyze_hooks


def create_hook_fns_train(start_grad_step,
                          num_grad_steps):

    hook_fns_frequencies = [
        # (5, utils.plot.hook_plot_model_community_detection),
        (0, hook_log_args),
        (5, hook_print_model_progress),
        (5, hook_write_scalars),
        (50, utils.plot.hook_plot_within_trial_stimuli_and_model_prob),
        (50, utils.plot.hook_plot_fraction_var_explained),
        (100, utils.plot.hook_plot_avg_model_prob_by_trial_within_block),
        (100, utils.plot.hook_plot_psychometric_curves),
        (100, utils.plot.hook_plot_model_weights),
        # (10, utils.plot.hook_plot_model_weights_gradients),
        (100, utils.plot.hook_plot_hidden_state_correlations),
        # (10, utils.plot.hook_plot_pca_hidden_state_activity_within_block),
        (100, utils.plot.hook_plot_pca_hidden_state_vector_fields),
        (100, utils.plot.hook_plot_pca_hidden_state_trajectories_within_block),
        # (10, utils.plot.hook_plot_hidden_state_projected_trajectories_controlled),
        (100, utils.plot.hook_plot_pca_hidden_state_fixed_points),
        # (10, utils.plot.hook_plot_psytrack_fit),
        # (10, utils.plot.hook_plot_hidden_to_hidden_jacobian_eigenvalues_complex_plane),
        (10000, hook_save_model),
    ]

    train_hooks = create_hook_fns_dict(
        hook_fns_frequencies=hook_fns_frequencies,
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    return train_hooks


def hook_log_args(hook_input):

    notes_dict = dict(
        model_str=hook_input['model'].model_str,
        model_kwargs=hook_input['model'].model_kwargs,
        batch_size=len(hook_input['envs']),
        env_loss_fn_str=hook_input['envs'][0].loss_fn_str,
        blocks_per_session=hook_input['envs'][0].blocks_per_session,
        trials_per_block_param=hook_input['envs'][0].trials_per_block_param,
        min_trials_per_block=hook_input['envs'][0].min_trials_per_block,
        max_trials_per_block=hook_input['envs'][0].max_trials_per_block,
        max_rnn_steps_per_trial=hook_input['envs'][0].max_rnn_steps_per_trial,
        possible_trial_strengths=hook_input['envs'][0].possible_trial_strengths,
        possible_trial_strengths_probs=hook_input['envs'][0].possible_trial_strengths_probs,
        block_side_probs=hook_input['envs'][0].block_side_probs,
        time_delay_penalty=hook_input['envs'][0].time_delay_penalty,
        seed=hook_input['seed'],
    )

    notes_file = os.path.join(
        hook_input['tensorboard_writer'].get_logdir(),
        'notes.json')
    f = open(notes_file, "w")
    f.write(json.dumps(notes_dict, indent=4, sort_keys=True))
    f.close()


def hook_print_model_progress(hook_input):
    print('Grad Step: {:5d}\tAvg Loss: {:6.3f}\tAvg Reward: {:6.3f}\tAvg RNN Steps/Trial: {:6.3f}'.format(
        hook_input['grad_step'],
        hook_input['avg_loss'],
        hook_input['avg_reward'],
        hook_input['avg_rnn_steps_per_trial']))


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
        model.description_str + ', grad_steps={:04d}.pt'.format(
            hook_input['grad_step']))

    torch.save(
        obj=save_dict,
        f=checkpoint_file_path)

    print('Saved model!')


def hook_write_scalars(hook_input):

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'loss_per_grad_step',
        scalar_value=hook_input['avg_loss'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'reward_per_grad_step',
        scalar_value=hook_input['avg_reward'],
        global_step=hook_input['grad_step'])

    hook_input['tensorboard_writer'].add_scalar(
        tag=hook_input['tag_prefix'] + 'rnn_steps_per_trial',
        scalar_value=hook_input['avg_rnn_steps_per_trial'],
        global_step=hook_input['grad_step'])

    # plot the variance, fraction of variance for the first 5 PCs (arbitrary cutoff)
    num_pcs_to_plot = 5
    for i in range(num_pcs_to_plot):

        hook_input['tensorboard_writer'].add_scalar(
            tag=hook_input['tag_prefix'] + f'pc/{i+1}_frac_variance_explained',
            scalar_value=hook_input['frac_variance_explained'][i],
            global_step=hook_input['grad_step'])

        hook_input['tensorboard_writer'].add_scalar(
            tag=hook_input['tag_prefix'] + f'pc/{i+1}_variance_explained',
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
