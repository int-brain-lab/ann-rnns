import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from psytrack.plot.analysisFunctions import makeWeightPlot
import scipy.cluster.hierarchy as spc
import seaborn as sns

import utils.analysis

# increase resolution
plt.rcParams["figure.dpi"] = 100.

# map for converting left and right to numeric -1, 1 and vice versa
side_string_map = {
    'left': -1,
    -1: 'left',
    'right': 1,
    1: 'right'
}

side_color_map = {
    'left': 'tab:orange',
    side_string_map['left']: 'tab:orange',
    'right': 'tab:blue',
    side_string_map['right']: 'tab:blue',
}


def hook_plot_avg_model_prob_by_trial_num_within_block(hook_input):
    trial_data = hook_input['run_envs_output']['trial_data']

    # drop block number 1
    # at least until we can figure out what to do with zero-initialized hidden state
    trial_data = trial_data[trial_data['stimuli_block_number'] != 1]

    # plot trial number within block (x) vs probability of correct response (y)
    mean_model_correct_action_prob_by_trial_num = trial_data.groupby(
        'trial_num_within_block')['model_correct_action_probs'].mean()
    sem_model_correct_action_prob_by_trial_num = trial_data.groupby(
        'trial_num_within_block')['model_correct_action_probs'].sem()

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.errorbar(
        x=mean_model_correct_action_prob_by_trial_num.index.values,
        y=mean_model_correct_action_prob_by_trial_num,
        yerr=sem_model_correct_action_prob_by_trial_num,
        alpha=0.8,
        ms=3)
    ax.set_ylim([0.4, 1.0])
    ax.set_xlim([0., 120.])
    ax.set_xlabel('Trial Number within Block')
    ax.set_ylabel('Average Model Probability for Correct Choice')
    fig.suptitle('Average Model Probability for Correct Choice by Trial within Block')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    hook_input['tensorboard_writer'].add_figure(
        tag='avg_model_prob_by_trial_num_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_state_dimensionality(hook_input):
    hidden_states = hook_input['run_envs_output']['hidden_states']
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    eigenvalues = utils.analysis.compute_eigenvalues_svd(matrix=hidden_states)
    frac_variance_explained = np.cumsum(eigenvalues / np.sum(eigenvalues))
    eigenvalue_index = np.arange(1, 1 + len(frac_variance_explained))

    plt.plot(eigenvalue_index,
             frac_variance_explained,
             'bo',
             alpha=0.8,
             ms=3)
    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle('Fraction of Variance Explained by Dimension')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Variance Explained')
    ax.set_ylim([0., 1.0])

    hook_input['tensorboard_writer'].add_figure(
        tag='var_explained_by_dimension',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_state_correlations(hook_input):
    trial_data = hook_input['run_envs_output']['trial_data']

    # drop block number 1
    # at least until we can figure out what to do with zero-initialized hidden state
    non_first_block_indices = trial_data['stimuli_block_number'] != 1

    # hidden states shape: (num trials, num layers, hidden dimension)
    hidden_states = hook_input['run_envs_output']['hidden_states'][non_first_block_indices]

    # reshape to (num trials, num layers * hidden dimension)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    hidden_state_correlations = np.corrcoef(hidden_states.T)

    # due to machine error, correlation matrix isn't exactly symmetric (typically has e-16 errors)
    # lets make it symmetric
    hidden_state_correlations = (hidden_state_correlations + hidden_state_correlations.T) / 2

    # compute pairwise distances
    pdist = spc.distance.pdist(hidden_state_correlations)
    linkage = spc.linkage(pdist, method='complete')
    labels = spc.fcluster(linkage, 0.5 * np.max(pdist), 'distance')
    indices = np.argsort(labels)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 8))
    recurrent_mask_str = hook_input['model'].model_kwargs['connectivity_kwargs']['recurrent_mask']
    fig.suptitle(f'Hidden State Correlations & Weights (Recurrent Mask: {recurrent_mask_str})')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    # plot hidden state correlations
    sns.heatmap(hidden_state_correlations[indices][:, indices],
                cmap='RdBu_r',
                ax=axes[0],
                vmin=-1.,
                vmax=1.,
                square=True,
                xticklabels=indices,
                yticklabels=indices,
                cbar_kws={'label': 'Correlation', 'shrink': 0.5})
    axes[0].set_title('Hidden Unit Correlations')
    axes[0].set_xlabel('Hidden Unit Number')
    axes[0].set_ylabel('Hidden Unit Number')
    axes[0].set_aspect("equal")  # ensures little squares don't become rectangles

    # plot recurrent matrix values
    sns.heatmap(hook_input['model'].core.weight_hh_l0.data.numpy()[indices][:, indices],
                cmap='RdBu_r',
                ax=axes[1],
                # vmin=-1.,
                # vmax=1.,
                square=True,
                xticklabels=indices,
                yticklabels=indices,
                cbar_kws={'label': 'Weight Strength', 'shrink': 0.5})
    axes[1].set_title('Recurrent Weight Strength')
    axes[1].set_xlabel('Hidden Unit Number')
    axes[1].set_ylabel('Hidden Unit Number')
    axes[1].set_aspect("equal")  # ensures little squares don't become rectangles

    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_state_correlations',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_state_projected_fixed_points(hook_input):
    displacement_norm_cutoff = 0.5

    # TODO: deduplicate with hook_plot_hidden_state_projected_vector_fields
    fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']

    num_stimuli = len(fixed_points_by_side_by_stimuli[1.0].keys())
    fig, axes = plt.subplots(nrows=num_stimuli,
                             ncols=3,
                             gridspec_kw={"width_ratios": [1, 1, 0.05]},
                             figsize=(12, 8),
                             sharex=True,
                             sharey=True)

    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for c, (side, fixed_points_by_stimuli_dict) in \
            enumerate(fixed_points_by_side_by_stimuli.items()):

        for r, (stimulus, fixed_points_dict) in enumerate(fixed_points_by_stimuli_dict.items()):

            num_grad_steps = fixed_points_dict['num_grad_steps']

            ax = axes[r, c]
            ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
            ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
            if r == 0:
                ax.set_title(f'Block Side: {side_string_map[side]}')
            elif r == num_stimuli - 1:
                ax.set_xlabel('Principal Component #1')

            if c == 0:
                ax.set_ylabel(f'Stimulus: {stimulus}')
            else:
                ax.set_yticks([], [])

            displacement_norms = fixed_points_dict['normalized_displacement_vector_norm']
            smallest_displacement_norm_indices = displacement_norms.argsort()
            smallest_displacement_norm_indices = smallest_displacement_norm_indices[
                displacement_norms[smallest_displacement_norm_indices] < displacement_norm_cutoff]

            try:

                x = fixed_points_dict['pca_final_sampled_hidden_states'][smallest_displacement_norm_indices, 0]
                y = fixed_points_dict['pca_final_sampled_hidden_states'][smallest_displacement_norm_indices, 1]
                colors = fixed_points_dict['normalized_displacement_vector_norm'][smallest_displacement_norm_indices]

                sc = ax.scatter(
                    x,
                    y,
                    c=colors,
                    vmin=0,
                    vmax=displacement_norm_cutoff,
                    s=1,
                    cmap='gist_rainbow')

                # emphasize the fixed point with smallest gradient
                sc = ax.scatter(
                    [x[0]],
                    [y[0]],
                    c=[colors[0]],
                    edgecolors='k',
                    vmin=0,
                    vmax=displacement_norm_cutoff,
                    cmap='gist_rainbow'
                )

            except IndexError:
                print('No fixed points below displacement norm cutoff')

    fig.suptitle(f'Fixed Points (Num Grad Steps = {num_grad_steps})')

    # merge the rightmost column for the colorbar
    gs = axes[0, 2].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(sc, cax=ax_colorbar)
    color_bar.set_label(r'$||h_t - RNN(h_t, s_t) ||_2$')
    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_state_projected_phase_space_fixed_points',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_state_projected_phase_space(hook_input):
    trial_data = hook_input['run_envs_output']['trial_data']

    # create possible color range
    max_block_len = max(trial_data.groupby(['env_num', 'stimuli_block_number']).size())
    color_range = np.arange(max_block_len)

    # separate by side bias
    fig, axes = plt.subplots(nrows=1,
                             ncols=3,
                             gridspec_kw={"width_ratios": [1, 1, 0.05]},
                             figsize=(12, 8))
    plt.suptitle(f'Model State Space (Projected)')
    for side, trial_data_preferred_side in trial_data.groupby('stimuli_preferred_sides'):

        if side_string_map[side] == 'left':
            ax = axes[0]
            ax.set_title(f'Left Biased Blocks')
            ax.set_ylabel('Principal Component #2')
        elif side_string_map[side] == 'right':
            ax = axes[1]
            ax.set_title(f'Right Biased Blocks')
            ax.set_yticks([], [])
        else:
            raise ValueError('Unknown side!')

        ax.set_xlabel('Principal Component #1')
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
        fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

        # separate again by block number
        for (env_num, block_number), trial_data_by_block \
                in trial_data_preferred_side.groupby(['env_num', 'stimuli_block_number']):
            block_indices = trial_data_by_block.index.values
            proj_hidden_states_block = hook_input['pca_hidden_states'][block_indices]
            trial_colors = color_range[:len(block_indices)]
            sc = ax.scatter(
                x=proj_hidden_states_block[:, 0],
                y=proj_hidden_states_block[:, 1],
                s=1,
                c=trial_colors)

    color_bar = fig.colorbar(sc, cax=axes[2])
    color_bar.set_label('Trial Number within Block')
    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_state_projected_phase_space',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_state_projected_vector_fields(hook_input):
    # TODO: deduplicate with hook_plot_hidden_state_projected_fixed_points

    trial_data = hook_input['run_envs_output']['trial_data']

    # hidden states shape: (num trials, num layers, hidden dimension)
    hidden_states = hook_input['run_envs_output']['hidden_states']

    vector_fields_by_side_by_stimuli = utils.analysis.compute_projected_hidden_state_vector_field(
        model=hook_input['model'],
        trial_data=trial_data,
        hidden_states=hidden_states)

    num_stimuli = len(vector_fields_by_side_by_stimuli[1.0].keys())

    fig, axes = plt.subplots(nrows=num_stimuli,
                             ncols=3,
                             gridspec_kw={"width_ratios": [1, 1, 0.05]},
                             figsize=(12, 8),
                             sharex=True,
                             sharey=True)
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for c, (side, vector_fields_by_stimuli_dict) in \
            enumerate(vector_fields_by_side_by_stimuli.items()):

        for r, (stimulus, vector_field_dict) in enumerate(vector_fields_by_stimuli_dict.items()):

            ax = axes[r, c]
            ax.set_xlim(vector_field_dict['xrange'][0], vector_field_dict['xrange'][1])
            ax.set_ylim(vector_field_dict['yrange'][0], vector_field_dict['yrange'][1])
            if r == 0:
                ax.set_title(f'Block Side: {side_string_map[side]}')
            elif r == num_stimuli - 1:
                ax.set_xlabel('Principal Component #1')

            if c == 0:
                ax.set_ylabel(f'Stimulus: {stimulus}')
                # ax.set_ylabel('Principal Component #2')
            else:
                ax.set_yticks([], [])

            vector_magnitude = np.linalg.norm(vector_field_dict['displacement_vector'], axis=1)

            qvr = ax.quiver(
                vector_field_dict['projected_sampled_hidden_states'][:, 0],
                vector_field_dict['projected_sampled_hidden_states'][:, 1],
                0.01 * vector_field_dict['displacement_vector'][:, 0] / vector_magnitude,
                0.01 * vector_field_dict['displacement_vector'][:, 1] / vector_magnitude,
                vector_magnitude,
                scale=.1,
                cmap='gist_rainbow')

    # merge the rightmost column for the colorbar
    gs = axes[0, 2].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(qvr, cax=ax_colorbar)
    color_bar.set_label(r'$||h_t - RNN(h_t, s_t) ||_2$')

    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_state_projected_phase_space_vector_field',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_state_projected_trajectories(hook_input):
    trial_data = hook_input['run_envs_output']['trial_data']

    # select only environment 0, first 8 blocks
    subset_trial_data = trial_data[(trial_data['env_num'] == 0) &
                                   (trial_data['stimuli_block_number'] < 12)]

    # create possible color range
    max_block_len = max(subset_trial_data.groupby(['env_num', 'stimuli_block_number']).size())

    # separate by side bias
    num_rows, num_cols = 3, 4
    fig, axes = plt.subplots(nrows=3,
                             ncols=4,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]},
                             figsize=(18, 12))
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    plt.suptitle(f'Model State Space (Projected) Trajectories')

    for block_num, trial_data_by_block in subset_trial_data.groupby('stimuli_block_number'):
        row, col = block_num // num_cols, block_num % num_cols
        ax = axes[row, col]
        ax.set_title(f'Block Num: {1 + block_num}')
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
        if row == num_rows - 1:
            ax.set_xlabel('Principal Component #1')
        if col == 0:
            ax.set_ylabel('Principal Component #2')

        block_indices = trial_data_by_block.index.values
        proj_hidden_states_block = hook_input['pca_hidden_states'][block_indices]
        # stimuli = np.round(trial_data_by_block['stimuli'].values, 1)
        segment_text = np.where(trial_data_by_block['actions_correct'], 'C', 'I')
        for i in range(len(block_indices) - 1):
            ax.plot(
                proj_hidden_states_block[i:i + 2, 0],
                proj_hidden_states_block[i:i + 2, 1],
                color=plt.cm.jet(i / max_block_len))
            ax.text(
                proj_hidden_states_block[i, 0],
                proj_hidden_states_block[i, 1],
                # str(stimuli[i]),
                segment_text[i])

    # TODO: add colobar without disrupting
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max_block_len))
    # color_bar = fig.colorbar(sm, cax=axes[-1])
    # color_bar.set_label('Trial Number within Block')
    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_state_projected_phase_space_trajectories',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_state_projected_trajectories_controlled(hook_input):
    trajectory_controlled_output = utils.analysis.compute_projected_hidden_state_trajectory_controlled(
        model=hook_input['model'],
        pca=hook_input['pca'])

    trial_data = trajectory_controlled_output['trial_data']
    max_block_len = max(trial_data.groupby(['env_num', 'stimuli_block_number']).size())

    fig, axes = plt.subplots(nrows=3,
                             ncols=4,  # 1 row, 3 columns
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]},
                             figsize=(18, 12))
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    plt.suptitle(f'Model State Space (Projected) Smooth Trajectories')

    for block_num, trial_data_by_block in trial_data.groupby('stimuli_block_number'):
        row, col = block_num // 4, block_num % 4  # hard coded for 2 rows, 4 columns
        ax = axes[row, col]
        ax.set_title(f'Block Num: {1 + block_num}')
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
        if row == 1:
            ax.set_xlabel('Principal Component #1')
        if col == 0:
            ax.set_ylabel('Principal Component #2')

        block_indices = trial_data_by_block.index.values
        proj_hidden_states_block = trajectory_controlled_output['projected_hidden_states'][block_indices]
        stimuli = np.round(trial_data_by_block['stimuli'].values, 1)
        for i in range(len(block_indices) - 1):
            ax.plot(
                proj_hidden_states_block[i:i + 2, 0],
                proj_hidden_states_block[i:i + 2, 1],
                color=plt.cm.jet(i / max_block_len))
            ax.text(
                proj_hidden_states_block[i + 1, 0],
                proj_hidden_states_block[i + 1, 1],
                str(stimuli[i]))

    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_state_projected_phase_space_trajectories_controlled',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_to_hidden_jacobian_eigenvalues_complex_plane(hook_input):
    fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']

    # plot each fixed point in phase space

    jacobians_by_side_by_stimuli = utils.analysis.compute_jacobians_by_side_by_stimuli(
        model=hook_input['model'],
        trial_data=hook_input['run_envs_output']['trial_data'],
        fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli)

    num_stimuli = len(fixed_points_by_side_by_stimuli[1.0].keys())
    fig, axes = plt.subplots(nrows=num_stimuli,
                             ncols=2,  # rows, cols
                             gridspec_kw={"width_ratios": [1, 1]},
                             figsize=(12, 8),
                             sharex=True,
                             sharey=True)
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    jacobian_colors = dict(
        hidden_to_hidden='tab:blue',
        stimuli_to_hidden='tab:orange',
        rewards_to_hidden='tab:green')

    for c, (side, jacobians_by_stimuli) in \
            enumerate(jacobians_by_side_by_stimuli.items()):

        for r, (stimulus, jacobians) in enumerate(jacobians_by_stimuli.items()):

            ax = axes[r, c]
            if r == 0:
                ax.set_title(f'Block Side: {side_string_map[side]}')
            elif r == num_stimuli - 1:
                ax.set_xlabel(r'$\Re(\lambda)$')

            if c == 0:
                ax.set_ylabel(r'$\Im(\lambda)$')

            for jacobian_name, jacobian in jacobians.items():

                if jacobian_name != 'hidden_to_hidden':
                    continue

                jacobian_eigvals = np.linalg.eigvals(jacobian)
                print(max(jacobian_eigvals))

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)

                sc = ax.scatter(
                    jacobian_eigvals.real,
                    jacobian_eigvals.imag,
                    c=jacobian_colors[jacobian_name],
                    s=2,
                    label=jacobian_name)

            ax.legend()

            # add circle
            circle = plt.Circle((0, 0), radius=1, color='k', fill=False)
            ax.add_patch(circle)

    fig.suptitle(f'Hidden to Hidden Jacobians\' Eigenvalues')
    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_to_hidden_jacobian_eigenvalues_complex_plane',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_hidden_to_hidden_jacobian_time_constants(hook_input):

    fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']

    # plot each fixed point in phase space

    jacobians_by_side_by_stimuli = utils.analysis.compute_jacobians_by_side_by_stimuli(
        model=hook_input['model'],
        trial_data=hook_input['run_envs_output']['trial_data'],
        fixed_points_by_side_by_stimuli=fixed_points_by_side_by_stimuli)

    num_stimuli = len(fixed_points_by_side_by_stimuli[1.0].keys())
    fig, axes = plt.subplots(nrows=num_stimuli,
                             ncols=2,  # rows, cols
                             gridspec_kw={"width_ratios": [1, 1]},
                             figsize=(12, 8),
                             sharex=True,
                             sharey=True)
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    jacobian_colors = dict(
        hidden_to_hidden='tab:blue',
        stimuli_to_hidden='tab:orange',
        rewards_to_hidden='tab:green')

    for c, (side, jacobians_by_stimuli) in \
            enumerate(jacobians_by_side_by_stimuli.items()):

        for r, (stimulus, jacobians) in enumerate(jacobians_by_stimuli.items()):

            ax = axes[r, c]
            if r == 0:
                ax.set_title(f'Block Side: {side_string_map[side]}')
            elif r == num_stimuli - 1:
                ax.set_xlabel('Eigenvalue Index')

            if c == 0:
                ax.set_ylabel(r'Time Constant ($\tau$)')

            for jacobian_name, jacobian in jacobians.items():

                if jacobian_name != 'hidden_to_hidden':
                    continue

                jacobian_eigvals = np.linalg.eigvals(jacobian)
                time_constants = np.sort(np.abs(1. / np.log(np.abs(jacobian_eigvals))))[::-1]
                eigvals_indices = np.arange(1, 1 + len(jacobian_eigvals))

                sc = ax.scatter(
                    eigvals_indices,
                    time_constants,
                    c=jacobian_colors[jacobian_name],
                    # s=2,
                    label=jacobian_name)

            ax.legend()

    fig.suptitle('Hidden to Hidden Jacobians\' Time Constants')
    # TODO understand why this produces such inconsistent plots
    hook_input['tensorboard_writer'].add_figure(
        tag='hidden_to_hidden_jacobian_time_constants',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_model_weights(hook_input):
    weights = dict(
        readout=hook_input['model'].readout.weight.data.numpy()
    )
    if hook_input['model'].model_str == 'rnn':
        weights['input'] = hook_input['model'].core.weight_ih_l0.data.numpy()
        weights['recurrent'] = hook_input['model'].core.weight_hh_l0.data.numpy()
    else:
        raise NotImplementedError

    # compute min, max
    vmin, vmax = np.inf, -np.inf
    for weight in weights.values():
        vmin = np.minimum(vmin, np.min(weight))
        vmax = np.maximum(vmax, np.max(weight))

    fig, axes = plt.subplots(nrows=1,
                             ncols=3,  # rows, cols
                             gridspec_kw={"width_ratios": [1, 1, 1]},
                             figsize=(12, 8))
    recurrent_mask_str = hook_input['model'].model_kwargs['connectivity_kwargs']['recurrent_mask']
    fig.suptitle(f'Model Weights (Recurrent Mask: {recurrent_mask_str})')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for i, (weight_str, weight_matrix) in enumerate(weights.items()):
        ax = axes[i]
        ax.set_title(f'{weight_str}')
        # ax.set_xlabel('Hidden Unit Number')
        # ax.set_ylabel('Hidden Unit Number')
        ax.set_aspect("equal")  # ensures little squares don't become rectangles
        sns.heatmap(weight_matrix, cmap='RdBu_r', square=True, ax=ax,
                         cbar_kws={'label': 'Weight Strength', 'shrink': 0.5}, vmin=vmin, vmax=vmax)

    hook_input['tensorboard_writer'].add_figure(
        tag='model_weights',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_model_weights_gradients(hook_input):
    model = hook_input['model']
    if model.model_str == 'rnn':
        recurrent_weight_grad = model.core.weight_hh_l0.grad.numpy()
    elif model.model_str == 'gru':
        raise NotImplementedError
    elif model.model_str == 'lstm':
        raise NotImplementedError
    else:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(recurrent_weight_grad, cmap='RdBu_r', square=True,
                     vmin=-0.1, vmax=0.1,
                     cbar_kws={'label': 'Gradient Value'})
    fig.suptitle('Recurrent Weight Matrix Gradients')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.invert_yaxis()
    ax.set_xlabel('Hidden Unit Number')
    ax.set_ylabel('Hidden Unit Number')
    ax.set_aspect("equal")  # ensures little squares don't become rectangles
    hook_input['tensorboard_writer'].add_figure(
        tag='recurrent_weight_matrix_gradients',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_psychometric_curves(hook_input):
    # drop block number 1
    # at least until we can figure out what to do with zero-initialized hidden state
    trial_data = hook_input['run_envs_output']['trial_data']
    trial_data = trial_data[trial_data['stimuli_block_number'] != 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    for preferred_side, preferred_side_group \
            in trial_data.groupby('stimuli_preferred_sides'):
        stimuli_strengths = preferred_side * (
                preferred_side_group['stimuli'] - preferred_side_group['stimuli_sides'])  # preferred_side

        ax.plot(stimuli_strengths,
                preferred_side_group['model_correct_action_probs'],
                marker='o',
                linestyle='',
                ms=1,
                alpha=0.8,
                label=side_string_map[preferred_side])

    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    fig.suptitle('Psychometric Curves')
    ax.set_xlabel('Stimulus Strength: Side * (Stimulus Value - Stimulus Mean)')
    ax.set_ylabel('Probability of Correct Choice')
    ax.set_ylim([0.4, 1.0])
    ax.set_xlim([-3.3, 3.3])
    ax.legend(numpoints=1, loc='best')
    hook_input['tensorboard_writer'].add_figure(
        tag='psychometric_curves',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)


def hook_plot_psytrack_fit(hook_input):
    trial_data = hook_input['run_envs_output']['trial_data']

    # drop block 1, keep only env 0
    keep_indices = (trial_data['env_num'] == 0) & (trial_data['stimuli_block_number'] != 1)
    subset_trial_data = trial_data[keep_indices]

    try:
        psytrack_fit_output = utils.analysis.compute_psytrack_fit(
            trial_data=subset_trial_data)
    except RuntimeError:
        # Factor is exactly singular. can occur if model is outputting only one action
        return

    # if error was encountered, just skip
    if psytrack_fit_output is None:
        return
    wMAP, credibleInt = psytrack_fit_output['wMAP'], psytrack_fit_output['credibleInt']

    # makeWeightPlot(
    #     wMode=wMAP,
    #     outData=psytrack_data,
    #     weights_dict=weights_dict,
    #     END=len(subset_trial_data),
    #     errorbar=credibleInt,
    #     perf_plot=True,
    #     bias_plot=True)

    # create subplots
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    num_trials_to_display = 500
    trial_num = np.arange(num_trials_to_display) + 1
    fig.suptitle(f'Bernoulli GLM Model (Psytrack by Roy & Pillow) (Num Points={len(subset_trial_data)})')
    axes[3].set_xlabel('Trial Number')

    # plot stimuli values
    axes[0].plot(
        trial_num,
        subset_trial_data['stimuli'].values[:num_trials_to_display],
        label='Stimulus Value')
    axes[1].set_ylabel('Stimulus Value')

    # plot block structure i.e. preferred side
    axes[1].plot(
        trial_num,
        subset_trial_data['stimuli_preferred_sides'].values[:num_trials_to_display],
        label='Block Preferred Side')
    axes[1].scatter(
        trial_num,
        1.05 * subset_trial_data['stimuli_sides'].values[:num_trials_to_display],
        alpha=0.8,
        s=1,
        c='tab:orange',
        label='Trial Correct Side')
    axes[1].set_ylabel('Block Preferred Side')
    axes[1].legend(loc="upper right")

    # plot weight time series
    stimuli_wMAP, reward_wMAP = wMAP[0, :num_trials_to_display], wMAP[1, :num_trials_to_display]
    stimuli_interval = credibleInt[0, :num_trials_to_display]
    reward_interval = credibleInt[1, :num_trials_to_display]
    axes[2].plot(
        trial_num,
        stimuli_wMAP,
        label='Stimulus Weight',
    )
    axes[2].fill_between(
        trial_num,
        stimuli_wMAP - 2 * stimuli_interval,
        stimuli_wMAP + 2 * stimuli_interval,
        alpha=0.8,
        linewidth=0)
    axes[2].set_ylabel('BernGLM Stimulus Weight')

    # add bias timeseries
    axes[3].plot(
        trial_num,
        reward_wMAP,
        label='Reward Weight')
    axes[3].fill_between(
        trial_num,
        reward_wMAP - 2 * reward_interval,
        reward_wMAP + 2 * reward_interval,
        alpha=0.8,
        linewidth=0)
    axes[3].set_ylabel('BernGLM Reward Weight')
    hook_input['tensorboard_writer'].add_figure(
        tag='psytrack_model',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True)
