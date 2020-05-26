import community
import logging
import networkx as nx
import networkx.algorithms.community
import networkx.drawing
import numpy as np
import os
import pandas as pd
from psytrack.helper.invBlkTriDiag import getCredibleInterval
from psytrack.hyperOpt import hyperOpt
from scipy.linalg import solve_discrete_lyapunov
import scipy.spatial
from scipy.stats import norm
from sklearn.decomposition.pca import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
import statsmodels.api as sm
import sys
import torch
import torch.autograd
import torch.optim

from utils.env import create_custom_worlds
from utils.run import run_envs


possible_stimuli = torch.DoubleTensor(
    [[1.2, 0.2],
     [0.2, 0.2],
     [0.2, 1.2],
     [0., 0.],
     [0., 0.],
     [0., 0.]])

possible_feedback = torch.DoubleTensor(
    [[-0.05],
     [-0.05],
     [-0.05],
     [-1],
     [0],
     [1]])


def add_analysis_data_to_hook_input(hook_input):
    # convert from shape (number of total time steps, num hidden layers, hidden size) to
    # shape (number of total time steps, num hidden layers * hidden size)

    # compute_behav_psychometric_comparison_between_model_and_mice(
    #     session_data=hook_input['session_data'])

    reshaped_hidden_states = hook_input['hidden_states'].reshape(
        hook_input['hidden_states'].shape[0], -1)

    hidden_states_pca_results = compute_model_hidden_states_pca(
        hidden_states=reshaped_hidden_states,
        model_readout_weights=hook_input['model'].readout.weight.data.numpy())

    hidden_states_jl_results = compute_model_hidden_states_jl(
        hidden_states=reshaped_hidden_states,
        model_readout_weights=hook_input['model'].readout.weight.data.numpy())

    model_block_readout_vectors_results = compute_model_block_readout_vectors(
        session_data=hook_input['session_data'],
        hidden_states=reshaped_hidden_states,
        pca_hidden_states=hidden_states_pca_results['pca_hidden_states'],
        pca=hidden_states_pca_results['pca'],
        trial_readout_vector=hidden_states_pca_results['trial_readout_vector'],
        pca_trial_readout_vector=hidden_states_pca_results['pca_trial_readout_vector'])

    model_task_aligned_states_results = compute_model_task_aligned_states(
        session_data=hook_input['session_data'],
        pca_hidden_states=hidden_states_pca_results['pca_hidden_states'],
        pca_trial_readout_vector=hidden_states_pca_results['pca_trial_readout_vector'],
        pca_block_readout_vector=model_block_readout_vectors_results['pca_block_readout_vector'])

    fixed_points_results = compute_model_fixed_points_by_stimulus_and_feedback(
        model=hook_input['model'],
        pca=hidden_states_pca_results['pca'],
        pca_xrange=hidden_states_pca_results['pca_xrange'],
        pca_yrange=hidden_states_pca_results['pca_yrange'],
        jlm=hidden_states_jl_results['jlm'],
        jlm_xrange=hidden_states_jl_results['jl_xrange'],
        jlm_yrange=hidden_states_jl_results['jl_yrange'],
        pca_hidden_states=hidden_states_pca_results['pca_hidden_states'],
        trial_readout_vector=hidden_states_pca_results['trial_readout_vector'],
        block_readout_vector=model_block_readout_vectors_results['block_readout_vector'],
        num_grad_steps=500)

    eigenvalues_svd_results = compute_eigenvalues(
        matrix=reshaped_hidden_states)

    reduced_dynamics_results = fit_reduced_dim_dynamics(
        session_data=hook_input['session_data'],
        pca=hidden_states_pca_results['pca'],
        task_aligned_hidden_states=model_task_aligned_states_results['task_aligned_hidden_states'])

    model_state_space_vector_fields_results = compute_model_state_space_vector_fields(
        session_data=hook_input['session_data'],
        pca_hidden_states=hidden_states_pca_results['pca_hidden_states'])

    optimal_observers_results = compute_optimal_observers(
        env=hook_input['envs'][0],
        session_data=hook_input['session_data'],
        rnn_steps_before_stimulus=hook_input['envs'][0].rnn_steps_before_stimulus,
        time_delay_penalty=hook_input['envs'][0].time_delay_penalty)

    mice_behavior_data_results = load_mice_behavioral_data(
        mouse_behavior_dir_path='data/ibl-data-may2020')

    # add results to hook_input
    result_dicts = [
        hidden_states_jl_results,
        hidden_states_pca_results,
        model_block_readout_vectors_results,
        model_task_aligned_states_results,
        fixed_points_results,
        eigenvalues_svd_results,
        reduced_dynamics_results,
        model_state_space_vector_fields_results,
        optimal_observers_results,
        mice_behavior_data_results]
    for result_dict in result_dicts:
        hook_input.update(result_dict)


def compute_eigenvalues(matrix):
    """
    matrix should have shape (num_samples, num_features)
    """
    pca = PCA()
    pca.fit(matrix)
    variance_explained = np.sort(pca.explained_variance_)[::-1]
    frac_variance_explained = np.cumsum(variance_explained / np.sum(variance_explained))

    eigenvalues_results = dict(
        variance_explained=variance_explained,
        frac_variance_explained=frac_variance_explained)

    return eigenvalues_results


# def compute_projected_hidden_state_trajectory_controlled(model,
#                                                          pca):
#     envs = create_custom_worlds(
#         num_envs=1,
#         blocks_per_session=12,
#         tensorboard_writer=None)
#     run_envs_output = run_envs(
#         model=model,
#         envs=envs)
#     hidden_states = run_envs_output['hidden_states']
#     projected_hidden_states = pca.transform(hidden_states.reshape(hidden_states.shape[0], -1))
#
#     trajectory_controlled_output = dict(
#         session_data=run_envs_output['session_data'],
#         hidden_states=hidden_states,
#         projected_hidden_states=projected_hidden_states,
#     )
#
#     return trajectory_controlled_output


def compute_behav_psychometric_comparison_between_model_and_mice(session_data):

    # only take consider last dt within a trial
    action_data = session_data.loc[
        session_data['action_taken'] == 1,
        ['block_side', 'signed_trial_strength', 'action_side']]

    # rescale from [-1, -1] to [0, 1]
    action_data['action_side'] = (1 + action_data['action_side']) / 2

    # normalize signed trial strengths to [-1, 1]
    action_data['signed_trial_strength'] /= action_data['signed_trial_strength'].max()

    # reset_index to add index (i.e. signed_trial_strength) as a column
    rnn_right_block_psychometric_data = action_data[action_data.block_side == 1.].groupby(
        ['signed_trial_strength']).agg(
        {'action_side': ['size', 'mean']})['action_side'].reset_index().to_numpy()

    mice_right_block_psychometric_data = np.array([
        [],
        [],
        []
    ])

    from submodules.iblanalysis.python.psychofit import mle_fit_psycho

    # threshold, slope, gamma1, gamma2
    # my guesses:
    # threshold (bias) is the contrast strength (x) at y=0.5
    # slope is the slope from y=0.25 to y=0.75
    # gamma is the lapse rate i.e. 1 - the upper right shoulder y value (or equivalently,
    #       the lower left shoulder y value - 0)
    rnn_params, rnn_mle = mle_fit_psycho(rnn_right_block_psychometric_data.T, 'erf_psycho')


def compute_model_fixed_points(model,
                               stimulus_val,
                               feedback_val,
                               initial_states,
                               num_grad_steps=10,
                               learning_rate=0.1,
                               momentum=0.9):
    # shape: (len(random subset indices, 1)
    feedback = torch.stack(len(initial_states) * [feedback_val], dim=0)

    # shape: (len(random subset indices), 1, 2)
    stimulus = torch.stack(len(initial_states) * [stimulus_val], dim=0).unsqueeze(1)

    # require grad to use fixed point finder i.e. minimize ||h_t - RNN(h_t, possible_stimulus)||
    final_states = initial_states.clone().requires_grad_(True)

    optimizer = torch.optim.SGD([final_states], lr=learning_rate, momentum=momentum)

    for i in range(num_grad_steps):

        # need to save hidden state for fixed point basin analysis
        if i == 1:
            second_states = model_forward_output['core_hidden'].clone()
        optimizer.zero_grad()
        model_forward_output = run_model_one_step(
            model=model,
            stimulus=stimulus,
            feedback=feedback,
            hidden_states=final_states)
        # non-LSTM shape: (session size, 1 time step, hidden state size)
        # LSTM shape: (session size, 1 time step, hidden state size, 2)
        model_forward_hidden_state = model_forward_output['core_hidden']
        displacement = model_forward_hidden_state - final_states
        # if LSTM, merge last two dimension
        # if len(displacement.shape) == 4:
        #     displacement = displacement.reshape(
        #         (len(displacement), 1, -1))
        #     model_forward_hidden_state = model_forward_hidden_state.reshape(
        #         (len(model_forward_hidden_state), 1, -1))
        displacement_norm = torch.norm(
            displacement,
            dim=(1, 2))
        loss = torch.mean(displacement_norm)
        loss.backward()
        optimizer.step()

    logging.info(f'Stimulus val: {stimulus_val.numpy()}\t\tFeedback val: {feedback_val.numpy()}')
    normalized_displacement_norm = torch.div(
        displacement_norm,
        torch.norm(final_states, dim=(1, 2)))
    logging.info(f'Minimum displacement norm: {torch.min(displacement_norm).item()}')

    model_fixed_points_results = dict(
        second_states=second_states.detach().numpy(),
        final_states=final_states.detach().numpy(),
        final_states_next_state=model_forward_hidden_state.detach().numpy(),
        displacement=displacement.detach().numpy(),
        displacement_norm=displacement_norm.detach().numpy(),
        normalized_displacement_norm=normalized_displacement_norm.detach().numpy(),
        stimulus=stimulus.detach().numpy(),
        feedback=feedback.detach().numpy())

    model_jacobians_results = compute_model_fixed_points_jacobians(
        model=model,
        stimulus=stimulus,
        feedback=feedback,
        fixed_points=final_states,
        displacement_norm=displacement_norm.detach().numpy())

    # add jacobians to result
    for key, value in model_jacobians_results.items():
        model_fixed_points_results[key] = value

    return model_fixed_points_results


def compute_model_fixed_points_basins_of_attraction(fixed_point_df):
    # discrete: http://automatica.dei.unipd.it/tl_files/utenti2/bof/Papers/NoteDiscreteLyapunov.pdf
    # continuous: http://sisdin.unipv.it/labsisdin/teaching/courses/ails/files/5-Lyapunov_theory_2_handout.pdf

    index = np.arange(len(fixed_point_df.groupby([
        'left_stimulus', 'right_stimulus', 'feedback'])))
    columns = [
        'left_stimulus',
        'right_stimulus',
        'feedback',
        'fixed_point_state',
        'pca_fixed_point_state',
        'fixed_point_displacement',
        'hidden_states_in_basin',
        'initial_pca_states_in_basin',
        'energy',
    ]
    fixed_points_basins_df = pd.DataFrame(
        np.nan,
        columns=columns,
        index=index,
        dtype=np.float16
    )

    for column in ['fixed_point_state', 'pca_fixed_point_state', 'hidden_states_in_basin',
                   'initial_pca_states_in_basin', 'energy']:
        fixed_points_basins_df[column] = fixed_points_basins_df[column].astype(np.object)

    for i, ((lstim, rstim, fdbk), fixed_point_subset_df) in enumerate(fixed_point_df.groupby([
        'left_stimulus', 'right_stimulus', 'feedback'], sort=False)):

        fixed_points_basins_df.at[i, 'left_stimulus'] = lstim
        fixed_points_basins_df.at[i, 'right_stimulus'] = rstim
        fixed_points_basins_df.at[i, 'feedback'] = fdbk

        logging.info(f'Lstim: {lstim}\t\tRstim: {rstim}\t\tFdbk: {fdbk}')

        # Jacobian and PCA Jacobian must be asymptotically stable
        stable_fp_df = fixed_point_subset_df[
            (fixed_point_subset_df['jacobian_hidden_stable'] == 1) &
            (fixed_point_subset_df['jacobian_pca_stable'] == 1)]

        if len(stable_fp_df) == 0:
            logging.info('No stable fixed points')
            continue

        # filter for MOST fixed point
        minimum_displacement = fixed_point_subset_df['displacement_norm'].min()
        minimum_displacement_index = fixed_point_subset_df['displacement_norm'].idxmin()
        fixed_point_state = fixed_point_subset_df.at[
            minimum_displacement_index, 'final_sampled_state']
        fixed_point_state_next_state = fixed_point_subset_df.at[
            minimum_displacement_index, 'final_sampled_state_next_state']
        pca_fixed_point_state = fixed_point_subset_df.at[
            minimum_displacement_index, 'final_pca_sampled_state']

        # solve the discrete Lyapunov equation
        Q = np.eye(len(fixed_point_state.flatten()))
        lambda_min_Q = np.min(np.linalg.eigvals(Q))
        A = np.array(fixed_point_subset_df.at[minimum_displacement_index, 'jacobian_hidden'])
        lambda_max_ATA = np.max(np.linalg.eigvals(A.T @ A))
        P = solve_discrete_lyapunov(a=A, q=Q)
        np.testing.assert_almost_equal(P, P.T)
        lambda_max_P = np.max(np.linalg.eigvals(P))

        # find parameter controlling basin of attraction
        # gamma should be positive
        gamma = np.max(np.roots([
            -lambda_max_P,
            -2 * lambda_max_P * np.sqrt(lambda_max_ATA),
            lambda_min_Q]))
        logging.info(f'Gamma: {gamma}')

        if gamma <= 0:
            logging.info('No positive roots found for gamma')
            continue

        initial_sampled_states = np.stack(
            fixed_point_subset_df['initial_sampled_state'].tolist(),
            axis=0)
        second_sampled_states = np.stack(
            fixed_point_subset_df['second_sampled_state'].tolist(),
            axis=0)

        # coordinate transform the system
        coord_transform_initial_states = initial_sampled_states - fixed_point_state
        coord_transform_second_states = second_sampled_states - fixed_point_state
        coord_transform_fixed_point_next_state = fixed_point_state_next_state - fixed_point_state

        # Let the system be x(t+1) = f(x(t)). Following the coordinate transform,
        # Taylor series about the fixed point x^* = 0. Then f(x) = f(x^*) + A x + g(x),
        # where A is the Jacobian evaluated at the fixed point.
        ts_linearized_component = np.einsum(
            'ijl,kl->ijk',
            coord_transform_initial_states,
            A)

        # g(x) is the Taylor Series remainder
        # because f(x^*) is not exactly zero, subtract it i.e. g(x) = f(x) - f(x^*) - A x
        ts_remainder = coord_transform_second_states - ts_linearized_component \
                       - coord_transform_fixed_point_next_state

        # select points such that ||g(x)|| < \gamma ||Ax||
        g_x_norm = np.linalg.norm(ts_remainder, axis=(1, 2))
        x_norm = np.linalg.norm(coord_transform_initial_states, axis=(1, 2))

        meets_criterion_indices = g_x_norm < (gamma * x_norm)
        logging.info(f'Fraction of points meeting criterion: {np.mean(meets_criterion_indices)}')

        if not np.any(meets_criterion_indices):
            continue

        # index_of_smallest_norm_meeting_criterion = fixed_point_subset_df.index[
        #     np.argmin(x_norm[meets_criterion_indices])]
        # radius of asymptotic stability
        # basin_radius = fixed_point_subset_df.loc[
        #     index_of_smallest_norm_meeting_criterion, 'displacement_pca_norm']
        hidden_states_in_basin = initial_sampled_states[meets_criterion_indices]
        initial_pca_states_in_basin = np.stack(fixed_point_subset_df.loc[
                                                   meets_criterion_indices, 'initial_pca_sampled_state'].values.tolist())
        within_region_coord_transform_states = coord_transform_initial_states[
            meets_criterion_indices].squeeze(1)
        # energy should be V(x) = 0.5*x^T P x
        # numpy is a bitch with matrix multiplication if the first axis has a batch
        # use einsum instead
        energies = 0.5 * np.einsum('ij,jk,ik->i',
                                   within_region_coord_transform_states,
                                   P,
                                   within_region_coord_transform_states)

        fixed_points_basins_subrows = {
            'fixed_point_state': fixed_point_state.astype(np.object),
            'pca_fixed_point_state': pca_fixed_point_state.astype(np.object),
            'fixed_point_displacement': minimum_displacement,
            'hidden_states_in_basin': hidden_states_in_basin.astype(np.object),
            'initial_pca_states_in_basin': initial_pca_states_in_basin.astype(np.object),
            'energy': energies.astype(np.object),
            # 'basin_radius': basin_radius,
        }
        for column, value in fixed_points_basins_subrows.items():
            fixed_points_basins_df.at[i, column] = value.tolist()

    return fixed_points_basins_df


def compute_model_fixed_points_by_stimulus_and_feedback(model,
                                                        pca,
                                                        pca_xrange,
                                                        pca_yrange,
                                                        jlm,
                                                        jlm_xrange,
                                                        jlm_yrange,
                                                        pca_hidden_states,
                                                        trial_readout_vector,
                                                        block_readout_vector,
                                                        num_grad_steps=100):
    assert num_grad_steps > 2

    sampled_states = sample_model_states_in_state_space(
        projection_obj=pca,
        xrange=pca_xrange,
        yrange=pca_yrange,
        pca_hidden_states=pca_hidden_states)
    initial_sampled_states = torch.from_numpy(np.expand_dims(sampled_states, axis=1))

    # calculate model's subjective belief
    subjective_trial_side = 2 * (np.dot(sampled_states, trial_readout_vector.T) > 0) - 1
    subjective_block_side = 2 * (np.dot(sampled_states, block_readout_vector.T) > 0) - 1

    # identify non-first block indices
    columns = [
        'subjective_block_side',
        'subjective_trial_side',
        'left_action_prob',
        'right_action_prob',
        'left_stimulus',
        'right_stimulus',
        'feedback',
        'initial_sampled_state',
        'second_sampled_state',  # necessary for fixed point basin analysis
        'initial_pca_sampled_state',
        'second_pca_sampled_state',
        'final_sampled_state',
        'final_sampled_state_next_state',
        'final_pca_sampled_state',
        'displacement',
        'displacement_pca',
        'displacement_norm',
        'displacement_pca_norm',
        'normalized_displacement_norm',
        'jacobian_hidden',
        'jacobian_hidden_sym',
        'jacobian_hidden_eigenspectrum',
        'jacobian_hidden_sym_stable',
    ]

    fixed_point_df = pd.DataFrame(
        np.nan,  # initialize all to nan
        index=np.arange(len(sampled_states) * len(possible_stimuli)),
        columns=columns,
        dtype=np.float16)
    for column in ['initial_sampled_state', 'second_sampled_state', 'initial_pca_sampled_state',
                   'second_pca_sampled_state', 'final_sampled_state', 'final_pca_sampled_state',
                   'jacobian_hidden', 'jacobian_hidden_sym', 'jacobian_hidden_eigenspectrum',
                   'displacement', 'displacement_pca', 'final_sampled_state_next_state']:
        fixed_point_df[column] = fixed_point_df[column].astype(object)

    logging.info(f'Computing fixed points using {num_grad_steps} gradient steps')
    for row_group, (feedback_val, stimulus_val) in enumerate(zip(possible_feedback, possible_stimuli)):

        model_fixed_points_results = compute_model_fixed_points(
            model=model,
            initial_states=initial_sampled_states,
            stimulus_val=stimulus_val,
            feedback_val=feedback_val,
            num_grad_steps=num_grad_steps)
        stimulus = model_fixed_points_results['stimulus']
        feedback = model_fixed_points_results['feedback']
        initial_pca_sampled_states = pca.transform(sampled_states)
        second_sampled_states = model_fixed_points_results['second_states']
        second_pca_sampled_states = pca.transform(
            second_sampled_states.reshape(len(second_sampled_states), -1))
        displacement_pca = second_pca_sampled_states - initial_pca_sampled_states
        final_sampled_states = model_fixed_points_results['final_states']
        final_pca_sampled_states = pca.transform(
            final_sampled_states.reshape(len(final_sampled_states), -1))

        fixed_point_subrows = {
            'subjective_block_side': subjective_block_side[:, 0],
            'subjective_trial_side': subjective_trial_side[:, 0],
            'left_stimulus': stimulus[:, 0, 0],
            'right_stimulus': stimulus[:, 0, 1],
            'feedback': feedback[:, 0],
            'initial_sampled_state': np.expand_dims(sampled_states, 1),
            'initial_pca_sampled_state': initial_pca_sampled_states,
            'second_sampled_state': second_sampled_states,
            'second_pca_sampled_state': second_pca_sampled_states,
            'final_sampled_state': final_sampled_states,
            'final_pca_sampled_state': final_pca_sampled_states,
            'final_sampled_state_next_state': model_fixed_points_results['final_states_next_state'],
            'displacement': model_fixed_points_results['displacement'],
            'displacement_pca': displacement_pca,
            'displacement_norm': model_fixed_points_results['displacement_norm'],
            'displacement_pca_norm': np.linalg.norm(displacement_pca, axis=1),
            'normalized_displacement_norm': model_fixed_points_results['normalized_displacement_norm'],
            'jacobian_hidden': model_fixed_points_results['jacobian_hidden'],
            'jacobian_hidden_stable': model_fixed_points_results['jacobian_hidden_stable'],
            'jacobian_hidden_sym': model_fixed_points_results['jacobian_hidden_sym'],
            'jacobian_hidden_eigenspectrum': model_fixed_points_results['jacobian_hidden_eigenspectrum'],
            'jacobian_hidden_sym_stable': model_fixed_points_results['jacobian_hidden_sym_stable'],
        }
        start_row = row_group * len(sampled_states)
        for column, value in fixed_point_subrows.items():
            # I can't believe I have to do this. As far as I can tell, Pandas
            # doesn't permit assigning numpy arrays to slices of dataframes.
            # Thus, I think I need to assign row by row
            for row in range(len(sampled_states)):
                fixed_point_df.at[start_row + row, column] = value[row]

    compute_model_fixed_points_jacobians_projected(
        fixed_point_df=fixed_point_df,
        pca=pca,
        jlm=jlm)

    fixed_points_basins_df = compute_model_fixed_points_basins_of_attraction(
        fixed_point_df=fixed_point_df)

    model_fixed_points_results = dict(
        fixed_point_df=fixed_point_df,
        fixed_points_basins_df=fixed_points_basins_df)

    return model_fixed_points_results


def compute_model_fixed_points_jacobians(model,
                                         stimulus,
                                         feedback,
                                         fixed_points,
                                         displacement_norm):
    model_jacobians_results = compute_model_hidden_state_jacobians(
        model=model,
        stimulus=stimulus,
        feedback=feedback,
        hidden_states=fixed_points)

    jacobians_hidden = model_jacobians_results['jacobian_hidden']
    model_jacobians_results['jacobian_hidden_eigenspectrum'] = np.sort(np.stack(
        [np.linalg.eigvals(jacobian_hidden)
         for jacobian_hidden in jacobians_hidden]),
        axis=1)
    model_jacobians_results['jacobian_hidden_stable'] = np.logical_and(
        np.all(np.abs(np.real(model_jacobians_results['jacobian_hidden_eigenspectrum'])) < 1,
               axis=1),
        displacement_norm < np.quantile(displacement_norm, .05)).astype(np.float16)

    jacobians_hidden_sym = 0.5 * (
            jacobians_hidden + np.transpose(jacobians_hidden, axes=(0, 2, 1)))
    model_jacobians_results['jacobian_hidden_sym'] = jacobians_hidden_sym
    model_jacobians_results['jacobian_hidden_sym_eigenspectrum'] = np.sort(np.stack(
        [np.linalg.eigvals(jacobian_hidden_sym)
         for jacobian_hidden_sym in jacobians_hidden_sym]),
        axis=1)
    model_jacobians_results['jacobian_hidden_sym_stable'] = np.logical_and(
        np.all(np.abs(np.real(model_jacobians_results['jacobian_hidden_sym_eigenspectrum'])) < 1,
               axis=1),
        np.quantile(displacement_norm, .05)).astype(np.float16)

    return model_jacobians_results


def compute_model_fixed_points_jacobians_projected(fixed_point_df,
                                                   pca,
                                                   jlm):
    # necessary because Pandas object series can't be directly convert to array
    jacobians = np.stack(fixed_point_df['jacobian_hidden'].values.tolist())
    jacobians_stable = fixed_point_df['jacobian_hidden_stable']
    logging.info(f'Fraction of stable Jacobians: {np.mean(jacobians_stable)}')

    displacement = np.stack(fixed_point_df['displacement'].values.tolist())
    jacobians_pca = pca.components_ @ jacobians @ pca.components_.T
    jacobians_pca_eigenspectra = np.sort(np.stack(
        [np.linalg.eigvals(jacobian_pca)
         for jacobian_pca in jacobians_pca]),
        axis=1)
    pca_displacement_norm = np.linalg.norm(
        np.einsum(
            'ijk,lk->ijl',
            displacement,
            pca.components_),
        axis=(1, 2))
    jacobians_pca_stable = np.logical_and(
        np.all(np.abs(np.real(jacobians_pca_eigenspectra)) < 1, axis=1),
        pca_displacement_norm < np.quantile(pca_displacement_norm, .05),
    ).astype(np.float16)
    logging.info(f'Fraction of stable PCA Jacobians: {np.mean(jacobians_pca_stable)}')

    jacobians_jlm = jlm.components_ @ jacobians @ jlm.components_.T
    jacobians_jlm_eigenspectra = np.sort(np.stack(
        [np.linalg.eigvals(jacobian_jlm)
         for jacobian_jlm in jacobians_jlm]),
        axis=1)
    jlm_displacement_norm = np.linalg.norm(
        np.einsum(
            'ijk,lk->ijl',
            displacement,
            jlm.components_),
        axis=(1, 2))
    jacobians_jlm_stable = np.logical_and(
        np.all(np.abs(np.real(jacobians_jlm_eigenspectra)) < 1, axis=1),
        jlm_displacement_norm < np.quantile(jlm_displacement_norm, .05),
    ).astype(np.float16)
    logging.info(f'Fraction of stable JL Jacobians: {np.mean(jacobians_jlm_stable)}')

    col_names = [
        'jacobian_pca', 'jacobian_pca_eigenspectrum', 'jacobian_pca_stable',
        'jacobian_jlm', 'jacobian_jlm_eigenspectrum', 'jacobian_jlm_stable']

    col_values = [
        jacobians_pca, jacobians_pca_eigenspectra, jacobians_pca_stable,
        jacobians_jlm, jacobians_jlm_eigenspectra, jacobians_jlm_stable
    ]

    for col_name, col_value in zip(col_names, col_values):
        if len(col_value.shape) > 1:
            col_value = col_value.astype(np.object)
        fixed_point_df[col_name] = pd.Series(
            col_value.tolist(), index=fixed_point_df.index)


# def compute_model_hidden_state_vector_field(model,
#                                             session_data,
#                                             hidden_states,
#                                             pca,
#                                             pca_hidden_states):
#     columns = [
#         'session_data_indices',
#         'rnn_step_within_session',
#         'block_index',
#         'trial_index',
#         'rnn_step_index',
#         'block_side',
#         'trial_side',
#         'trial_strength',
#         'left_stimulus',
#         'right_stimulus',
#         'reward',
#         'correct_action_prob',
#         'left_action_prob',
#         'right_action_prob',
#         'hidden_state',
#         'pca_hidden_state',
#         'next_hidden_state',
#         'pca_next_hidden_state',
#         'hidden_state_difference']
#
#     vector_fields_df = pd.DataFrame(
#         columns=columns,
#         dtype=np.float16)
#
#     # enable storing hidden states in the dataframe.
#     # need to make the column have type object to doing so possible
#     for column in ['hidden_state', 'pca_hidden_state', 'next_hidden_state',
#                    'pca_next_hidden_state', 'hidden_state_difference']:
#         vector_fields_df[column] = vector_fields_df[column].astype(object)
#
#     # sample subset of indices
#     random_subset_indices = np.random.choice(
#         session_data.index,
#         replace=False,
#         size=min(1000, len(session_data)))
#
#     sampled_hidden_states = hidden_states[random_subset_indices]
#     sampled_pca_hidden_states = pca_hidden_states[random_subset_indices]
#
#     for feedback, stimulus in product(possible_feedback, possible_stimuli):
#         # shape: (len(random subset indices, 1)
#         feedback = torch.stack(len(random_subset_indices) * [feedback], dim=0)
#
#         # shape: (len(random subset indices), 1, 2)
#         stimulus = torch.stack(len(random_subset_indices) * [stimulus], dim=0).unsqueeze(1)
#
#         model_forward_output = run_model_one_step(
#             model=model,
#             stimulus=stimulus,
#             feedback=feedback,
#             hidden_states=torch.from_numpy(sampled_hidden_states))
#
#         next_sampled_hidden_states = model_forward_output['core_hidden'].detach().numpy()
#
#         next_sampled_pca_hidden_states = pca.transform(
#             next_sampled_hidden_states.reshape(len(random_subset_indices), -1))
#
#         displacement_vector = next_sampled_pca_hidden_states - sampled_pca_hidden_states
#
#         vector_fields_subrows_df = dict(
#             displacement_vector=displacement_vector,
#             random_subset_indices=random_subset_indices,
#             sampled_hidden_states=sampled_hidden_states,
#             sampled_pca_hidden_states=sampled_pca_hidden_states,
#             next_sampled_hidden_states=next_sampled_hidden_states,
#             next_sampled_pca_hidden_states=next_sampled_pca_hidden_states)
#
#         vector_fields_subrows_df = pd.DataFrame(vector_fields_subrows_df)
#
#         vector_fields_df = pd.concat((vector_fields_df, vector_fields_subrows_df))
#
#     return vector_fields_df


def compute_model_hidden_states_jl(hidden_states,
                                   model_readout_weights):
    # ensure hidden states have 2 dimensions
    assert len(hidden_states.shape) == 2
    jlm = GaussianRandomProjection(n_components=2)
    jlm.fit(hidden_states)

    # ensure right trial vector points to the right
    trial_readout_vector = model_readout_weights[-1, np.newaxis, :]
    trial_readout_vector /= np.linalg.norm(trial_readout_vector)
    jl_trial_readout_vector = jlm.transform(trial_readout_vector)[0]
    jl_trial_readout_vector /= np.linalg.norm(jl_trial_readout_vector)

    right_trial_vector_points_right = np.dot(
        jl_trial_readout_vector,
        [1., 0]) > 0
    if not right_trial_vector_points_right:
        logging.info('Swapped JL right readout vector direction')
        jlm.components_[0, :] *= -1.
        jl_trial_readout_vector = jlm.transform(trial_readout_vector)[0]
        jl_trial_readout_vector /= np.linalg.norm(jl_trial_readout_vector)
    else:
        logging.info('Did not swap JL right readout vector direction')

    jl_hidden_states = jlm.transform(hidden_states)

    min_x, max_x = min(jl_hidden_states[:, 0]), max(jl_hidden_states[:, 0])
    min_y, max_y = min(jl_hidden_states[:, 1]), max(jl_hidden_states[:, 1])

    hidden_states_pca_results = dict(
        jl_hidden_states=jl_hidden_states,
        jl_xrange=(min_x, max_x),
        jl_yrange=(min_y, max_y),
        jlm=jlm,
        trial_readout_vector=trial_readout_vector,
        jl_trial_readout_vector=jl_trial_readout_vector)

    return hidden_states_pca_results


def compute_model_hidden_state_jacobians(model,
                                         stimulus,
                                         feedback,
                                         hidden_states):
    num_layers = model.model_kwargs['core_kwargs']['num_layers']
    hidden_size = model.model_kwargs['core_kwargs']['hidden_size']
    num_basis_vectors = num_layers * hidden_size
    unit_basis_vectors = torch.eye(n=num_basis_vectors).reshape(
        num_basis_vectors, num_layers, hidden_size)

    # ensure we can differentiate w.r.t. inputs
    input_vars = [stimulus, feedback, hidden_states]

    for input_var in input_vars:
        input_var.requires_grad_(True)

    # name, jacobian w.r.t. this input, reshape size
    jacobian_specifications = [
        ('hidden', hidden_states, (num_basis_vectors, num_layers * hidden_size)),
        ('stimulus', stimulus, (num_basis_vectors, 1)),
        ('feedback', feedback, (num_basis_vectors, 1))
    ]

    model_jacobians_results = dict()
    for name, inputs, reshape_size in jacobian_specifications:

        # ensure all gradients zeroed
        for input_var in input_vars:
            if input_var.grad is not None:
                input_var.grad.zero_()

        model_forward_output = run_model_one_step(
            model=model,
            stimulus=stimulus,
            feedback=feedback,
            hidden_states=hidden_states)

        jacobian_components = []
        for unit_vector in unit_basis_vectors:
            jacobian_component = torch.autograd.grad(
                outputs=model_forward_output['core_output'],
                inputs=inputs,
                grad_outputs=torch.stack([unit_vector] * len(inputs)),  # repeat batch_size times
                retain_graph=True)[0]
            jacobian_component = torch.mean(jacobian_component, dim=(1,))
            jacobian_components.append(jacobian_component)
        jacobian = torch.stack(jacobian_components, dim=1)

        def model_wrapper(hidden_states_for_jacobian):
            return run_model_one_step(
                model=model,
                stimulus=stimulus,
                feedback=feedback,
                hidden_states=hidden_states_for_jacobian)['core_output']

        # jacobian2 = torch.autograd.functional.jacobian(
        #     func=model_wrapper,
        #     inputs=hidden_states)

        # because feedback has dim 2 instead of dim 3, the jacobian will be shape (hidden dim, )
        # add an extra dimension for consistency with other two Jacobians
        if name == 'feedback':
            jacobian = torch.unsqueeze(jacobian, dim=2)

        model_jacobians_results['jacobian_' + name] = jacobian.detach().numpy()

    # ensures no problems arise later
    for input_var in input_vars:
        input_var.requires_grad_(False)

    return model_jacobians_results


def compute_model_hidden_states_pca(hidden_states,
                                    model_readout_weights):
    # ensure hidden states have 2 dimensions
    assert len(hidden_states.shape) == 2
    pca = PCA(n_components=2)
    pca.fit(hidden_states)

    # ensure right trial vector points to the right
    trial_readout_vector = model_readout_weights[-1, np.newaxis, :]
    trial_readout_vector /= np.linalg.norm(trial_readout_vector)
    pca_trial_readout_vector = pca.transform(trial_readout_vector)[0]
    pca_trial_readout_vector /= np.linalg.norm(pca_trial_readout_vector)

    right_trial_vector_points_right = np.dot(
        pca_trial_readout_vector,
        [1., 0]) > 0
    if not right_trial_vector_points_right:
        logging.info('Swapped PCA right readout vector direction')
        pca.components_[0, :] *= -1.
        pca_trial_readout_vector = pca.transform(trial_readout_vector)[0]
        pca_trial_readout_vector /= np.linalg.norm(pca_trial_readout_vector)
    else:
        logging.info('Did not swap PCA right readout vector direction')

    pca_hidden_states = pca.transform(hidden_states)

    min_x, max_x = min(pca_hidden_states[:, 0]), max(pca_hidden_states[:, 0])
    min_y, max_y = min(pca_hidden_states[:, 1]), max(pca_hidden_states[:, 1])

    hidden_states_pca_results = dict(
        pca_hidden_states=pca_hidden_states,
        pca_xrange=(min_x, max_x),
        pca_yrange=(min_y, max_y),
        pca=pca,
        trial_readout_vector=trial_readout_vector,
        pca_trial_readout_vector=pca_trial_readout_vector)

    return hidden_states_pca_results


def compute_model_block_readout_vectors(session_data,
                                        hidden_states,
                                        pca_hidden_states,
                                        pca,
                                        trial_readout_vector,
                                        pca_trial_readout_vector):

    # transform from {-1, 1} to {0, 1}
    block_sides = (1 + session_data.block_side.values) / 2

    # do classification twice, one in high dimension, once in PCA space
    names = ['full', 'pca']
    regressors = [hidden_states, pca_hidden_states]
    classifier_accuracy = dict()
    for name, regressor in zip(names, regressors):
        train_regressor, test_regressor, train_block_sides, \
        test_block_sides = train_test_split(
            regressor,
            block_sides,
            test_size=.33)

        logistic_regression = sm.Logit(
            endog=train_block_sides,
            exog=train_regressor)
        logistic_regression_result = logistic_regression.fit()

        # compute accuracy of classifier
        predicted_test_block_sides = logistic_regression_result.predict(test_regressor)
        block_classifier_accuracy = np.mean(test_block_sides == np.round(predicted_test_block_sides))

        classifier_accuracy[name] = block_classifier_accuracy

    session_data['classifier_block_side'] = 2. * logistic_regression_result.predict(pca_hidden_states) - 1.

    # select RIGHT block side readout vector
    pca_block_readout_vector = logistic_regression_result.params
    pca_block_readout_vector /= np.linalg.norm(pca_block_readout_vector)
    block_readout_vector = np.expand_dims(pca.inverse_transform(pca_block_readout_vector), [0])
    block_readout_vector /= np.linalg.norm(block_readout_vector)

    radians_btwn_trial_block_vectors = np.arccos(np.dot(
        block_readout_vector.flatten(),
        trial_readout_vector.flatten()))

    degrees_btwn_trial_block_vectors = round(
        180 * radians_btwn_trial_block_vectors / np.pi)

    logging.info(f'Degrees between vectors: {degrees_btwn_trial_block_vectors}')

    radians_btwn_pca_trial_block_vectors = np.arccos(np.dot(
        pca_block_readout_vector / np.linalg.norm(pca_block_readout_vector),
        pca_trial_readout_vector / np.linalg.norm(pca_trial_readout_vector)))

    degrees_btwn_pca_trial_block_vectors = round(
        180 * radians_btwn_pca_trial_block_vectors / np.pi)

    logging.info(f'Degrees between PCA vectors: {degrees_btwn_pca_trial_block_vectors}')

    block_readout_weights_results = dict(
        block_readout_vector=block_readout_vector,
        pca_block_readout_vector=pca_block_readout_vector,
        radians_btwn_pca_trial_block_vectors=radians_btwn_pca_trial_block_vectors,
        degrees_btwn_pca_trial_block_vectors=degrees_btwn_pca_trial_block_vectors,
        full_block_classifier_accuracy=classifier_accuracy['full'],
        pca_block_classifier_accuracy=classifier_accuracy['pca'])

    return block_readout_weights_results


def compute_model_state_space_vector_fields(session_data,
                                            pca_hidden_states):
    model_state_space_vector_fields = pd.DataFrame(
        columns=['left_stimulus', 'right_stimulus', 'feedback',
                 'pca_hidden_states_pre', 'pca_hidden_states_post',
                 'displacement_pca', 'displacement_pca_norm'],
        dtype=np.float16)
    for column in ['pca_hidden_states_pre', 'pca_hidden_states_post', 'displacement_pca', 'displacement_pca_norm']:
        model_state_space_vector_fields[column] = model_state_space_vector_fields.astype(np.object)

    for row_group, (feedback_val, stimulus_val) in enumerate(zip(possible_feedback, possible_stimuli)):

        if feedback_val.item() == 0 and torch.all(stimulus_val == 0.):  # blank dts
            task_condition_rows = (session_data.reward == 0.) & \
                                  (session_data.left_stimulus == 0) & \
                                  (session_data.right_stimulus == 0) & \
                                  (session_data.rnn_step_index == 1.)
        elif feedback_val.item() != 0. and torch.all(stimulus_val == 0):  # feedback dt
            task_condition_rows = (session_data.reward.shift(1) == feedback_val.item())
            # make sure last value is False to prevent indexing issues on last dt
            task_condition_rows[len(task_condition_rows) - 1] = False
        else:  # equal stimulus, strong left, or strong right
            task_condition_rows = (session_data.reward.shift(1) == feedback_val.item()) &\
                                  (stimulus_val[0].item() - 0.15 <= session_data.left_stimulus) & \
                                  (session_data.left_stimulus <= stimulus_val[0].item() + 0.15) & \
                                  (stimulus_val[1].item() - 0.15 <= session_data.right_stimulus) & \
                                  (session_data.right_stimulus <= stimulus_val[1].item() + 0.15)

        task_condition_index = task_condition_rows.index[task_condition_rows]
        # if too many, downsample
        if len(task_condition_index) > 1000:
            task_condition_index = np.random.choice(
                task_condition_index,
                size=1000,
                replace=False)

        pca_hidden_states_pre = pca_hidden_states[task_condition_index - 1]
        pca_hidden_states_post = pca_hidden_states[task_condition_index]
        displacement_pca = pca_hidden_states_post - pca_hidden_states_pre
        displacement_pca_norm = np.linalg.norm(displacement_pca, axis=1)

        model_state_space_vector_fields_subrows = dict(
            left_stimulus=stimulus_val[0].item(),
            right_stimulus=stimulus_val[1].item(),
            feedback=feedback_val.item(),
            pca_hidden_states_pre=pca_hidden_states_pre,
            pca_hidden_states_post=pca_hidden_states_post,
            displacement_pca=displacement_pca,
            displacement_pca_norm=displacement_pca_norm)

        model_state_space_vector_fields.loc[
            len(model_state_space_vector_fields)] = model_state_space_vector_fields_subrows

    model_state_space_vector_fields_results = dict(
        model_state_space_vector_fields=model_state_space_vector_fields)

    return model_state_space_vector_fields_results


def compute_model_task_aligned_states(session_data,
                                      pca_hidden_states,
                                      pca_trial_readout_vector,
                                      pca_block_readout_vector):

    task_aligned_directions = np.stack([
        pca_trial_readout_vector,
        pca_block_readout_vector])
    task_aligned_hidden_states = pca_hidden_states @ task_aligned_directions.T

    session_data['magn_along_block_vector'] = np.dot(
        task_aligned_hidden_states,
        pca_block_readout_vector.flatten())

    session_data['magn_along_trial_vector'] = np.dot(
        task_aligned_hidden_states,
        pca_trial_readout_vector.flatten())

    model_task_aligned_states_results = dict(
        task_aligned_hidden_states=task_aligned_hidden_states
    )

    return model_task_aligned_states_results


def compute_model_weights_directed_graph(model):
    weights = dict(
        input=model.core.weight_ih_l0.data.numpy(),
        recurrent=model.core.weight_hh_l0.data.numpy(),
        readout=model.readout.weight.data.numpy()
    )

    input_num_units = weights['input'].shape[1]
    recurrent_num_units = weights['recurrent'].shape[0]
    readout_num_units = weights['readout'].shape[0]
    total_num_units = input_num_units + recurrent_num_units + readout_num_units
    total_weight_matrix = np.zeros(shape=(total_num_units, total_num_units))

    # add weight matrices
    total_weight_matrix[:input_num_units, input_num_units:input_num_units + recurrent_num_units] = \
        weights['input'].T
    total_weight_matrix[input_num_units:input_num_units + recurrent_num_units,
    input_num_units:input_num_units + recurrent_num_units] = \
        weights['recurrent'].T
    total_weight_matrix[input_num_units:input_num_units + recurrent_num_units, input_num_units + recurrent_num_units:] = \
        weights['readout'].T

    model_graph = nx.convert_matrix.from_numpy_matrix(
        A=total_weight_matrix,
        create_using=nx.DiGraph)

    return model_graph


def compute_model_weights_community_detection(model):
    model_weights_directed_graph = compute_model_weights_directed_graph(model=model)
    model_graph_numpy = nx.to_numpy_array(model_weights_directed_graph)
    np.save('model_weights_directed_graph.npy', model_graph_numpy)
    partition = community.best_partition(model_weights_directed_graph)

    networkx.drawing.draw(
        model_weights_directed_graph,
        arrows=True)


def compute_optimal_observers(env,
                              session_data,
                              time_delay_penalty,
                              rnn_steps_before_stimulus):

    compute_optimal_block_side_inference(
        session_data=session_data,
        env=env)

    compute_optimal_trial_side_inference(
        session_data=session_data,
        env=env)

    # compute_optimal_coupled_observer(
    #     session_data=session_data)

    optimal_prob_correct_after_num_obs_blockless, optimal_prob_correct_after_num_obs_blockless_by_trial_strength = \
        compute_optimal_prob_correct_blockless(
            session_data=session_data,
            rnn_steps_before_stimulus=rnn_steps_before_stimulus)

    optimal_reward_rate_after_num_obs_blockless, optimal_reward_rate_after_num_obs_blockless_by_trial_strength = \
        compute_optimal_reward_rate_blockless(
            optimal_prob_correct_after_num_obs_blockless=optimal_prob_correct_after_num_obs_blockless,
            optimal_prob_correct_after_num_obs_blockless_by_trial_strength=optimal_prob_correct_after_num_obs_blockless_by_trial_strength,
            time_delay_penalty=time_delay_penalty)

    optimal_observers_results = dict(
        optimal_prob_correct_after_num_obs_blockless=optimal_prob_correct_after_num_obs_blockless,
        optimal_prob_correct_after_num_obs_blockless_by_trial_strength=optimal_prob_correct_after_num_obs_blockless_by_trial_strength,
        optimal_reward_rate_after_num_obs_blockless=optimal_reward_rate_after_num_obs_blockless,
        optimal_reward_rate_after_num_obs_blockless_by_trial_strength=optimal_reward_rate_after_num_obs_blockless_by_trial_strength)

    return optimal_observers_results


def compute_optimal_coupled_observer(session_data):

    # see https://github.com/bayespy/bayespy/issues/28

    non_blank_data = session_data[(session_data.left_stimulus != 0) &
                                  (session_data.right_stimulus != 0)]

    from bayespy.nodes import Categorical, CategoricalMarkovChain, Dirichlet,\
        Gaussian, Mixture, Wishart

    num_latent_variables = 4
    initial_state_probs = Dirichlet(np.array([10, 1, 10, 1]))

    transition_probs = Dirichlet(10*np.array([
        [0.5, 0.5, 0.98*0.2, 0.02*0.8],  # s_n = L, b_n = L
        [0.02*0.8, 0.98*0.2, 0.02*0.2, 0.98*0.8],  # s_n = L, b_n = R
        [0.98*0.2, 0.02*0.2, 0.98, 0.02],  # s_n = R, b_n = L
        [0.02*0.8, 0.98*0.2, 0.02, 0.98],  # s_n = R, b_n = R
    ]))

    latents = CategoricalMarkovChain(
        pi=initial_state_probs,
        A=transition_probs,
        states=len(non_blank_data))

    # approximate observation as Gaussian
    mu_est = Gaussian(np.zeros(1), 1e-5*np.identity(1), plates=(num_latent_variables,))
    Lambda_est = Wishart(1, np.identity(1), plates=(num_latent_variables,))

    mu_est = Gaussian(
        np.array([]),
        1e-5 * np.identity(1),
        plates=(num_latent_variables,))
    Lambda_est = Wishart(1, np.identity(1), plates=(num_latent_variables,))

    observations = Mixture(latents, Gaussian, mu_est, Lambda_est)

    diff_obs = non_blank_data['right_stimulus'] - non_blank_data['left_stimulus']
    # reshape to (number of non-blank dts, 1)
    diff_obs = np.expand_dims(diff_obs.values, axis=1)
    observations.observe(diff_obs)

    # Random initialization to break the symmetry
    # transition_probs.initialize_from_random()
    mu_est.initialize_from_random()

    # Reasonable initialization for Lambda
    Lambda_est.initialize_from_value(np.identity(1))

    from bayespy.inference import VB
    Q = VB(observations, latents, mu_est, Lambda_est, transition_probs, initial_state_probs)

    # use deterministic annealing to reduce sensitivity to initial conditions
    # https://www.bayespy.org/user_guide/advanced.html#deterministic-annealing
    beta = 0.1
    while beta < 1.0:
        beta = min(beta * 1.5, 1.0)
        Q.set_annealing(beta)
        Q.update(repeat=250, tol=1e-4)

    # recover transition probabilities by dividing by row sum
    initial_state_probs_posterior = Categorical(initial_state_probs).get_moments()[0]
    transition_probs_posterior = Categorical(transition_probs).get_moments()[0]
    logging.info('Crippled Bayesian Observer Transition Parameters:')
    logging.info(str(transition_probs_posterior))



def compute_optimal_block_side_inference(session_data,
                                         env):

    initial_state_probs = np.array([
        0.5, 0.5])

    transition_probs = np.array([
        [0.98, 0.02],
        [0.02, 0.98]])

    emission_probs = np.array([
        [0.8, 0.2],
        [0.2, 0.8]])

    trial_end_data = session_data[session_data.trial_end == 1.]
    latent_conditional_probs = np.zeros(shape=(len(trial_end_data), 2))
    trial_sides = ((1 + trial_end_data.trial_side.values) / 2).astype(np.int)

    # joint probability p(x_1, y_1)
    curr_joint_prob = np.multiply(
        emission_probs[trial_sides[0], :],
        initial_state_probs)

    for i, trial_side in enumerate(trial_sides[:-1]):
        # normalize to get P(b_n | s_{<=n})
        # np.sum(curr_joint_prob) is marginalizing over b_{n} i.e. \sum_{b_n} P(b_n, s_n |x_{<=n-1})
        curr_latent_conditional_prob = curr_joint_prob / np.sum(curr_joint_prob)
        latent_conditional_probs[i] = curr_latent_conditional_prob

        # P(y_{t+1}, x_{t+1} | x_{<=t})
        curr_joint_prob = np.multiply(
            emission_probs[trial_sides[i + 1], :],
            np.matmul(transition_probs, curr_latent_conditional_prob))

    # right block posterior, right block prior
    session_data['optimal_block_posterior_right'] = np.nan
    session_data.loc[trial_end_data.index, 'optimal_block_posterior_right'] = latent_conditional_probs[:, 1]
    session_data['optimal_block_prior_right'] = session_data['optimal_block_posterior_right'].shift(1)

    # right stimulus prior
    session_data['optimal_stimulus_prior_right'] = np.nan
    block_prior_indices = ~pd.isna(session_data['optimal_block_prior_right'])
    optimal_stimulus_prior_right = np.matmul(latent_conditional_probs[:-1, :], emission_probs.T)
    session_data.loc[block_prior_indices, 'optimal_stimulus_prior_right'] = optimal_stimulus_prior_right[:, 1]

    # manually specify that first block prior and first stimulus prior should be 0.5
    # before evidence, this is the correct prior
    session_data.loc[0, 'optimal_block_prior_right'] = 0.5
    session_data.loc[0, 'optimal_stimulus_prior_right'] = 0.5


def compute_optimal_trial_side_inference(session_data,
                                         env):

    strength_means = np.sort(session_data.signed_trial_strength.unique())
    prob_mu = env.possible_trial_strengths_probs

    # P(mu_n | s_n) as a matrix with shape (2 * number of stimulus strengths - 1, 2)
    # - 1 is for stimulus strength 0, which both stimulus sides can generate
    prob_mu_given_stim_side = np.zeros(shape=(len(strength_means), 2))
    prob_mu_given_stim_side[:len(prob_mu), 0] = prob_mu[::-1]
    prob_mu_given_stim_side[len(prob_mu)-1:, 1] = prob_mu

    diff_obs = session_data['right_stimulus'] - session_data['left_stimulus']

    session_data['optimal_stimulus_posterior_right'] = np.nan
    for (session_idx, block_idx, trial_idx), trial_data in session_data.groupby([
        'session_index', 'block_index', 'trial_index']):

        optimal_stimulus_prior_right = trial_data['optimal_stimulus_prior_right'].iloc[0]
        optimal_stim_prior = np.array([1 - optimal_stimulus_prior_right, optimal_stimulus_prior_right])

        # P(\mu_n, s_n | history) = P(\mu_n | s_n) P(s_n | history)
        # shape = (# of possible signed stimuli strengths, num trial sides)
        stim_side_strength_joint_prob = np.einsum(
            'ij,j->ij',
            prob_mu_given_stim_side,
            optimal_stim_prior)

        # exclude blank dts
        dt_indices = trial_data.iloc[env.rnn_steps_before_stimulus:].index
        trial_diff_obs = diff_obs[trial_data.index].values[
                         env.rnn_steps_before_stimulus:]

        # P(o_t | \mu_n, s_n) , also = P(o_t | \mu_n)
        # shape = (num of observations, # of possible signed stimuli strengths)
        individual_diff_obs_likelihood = scipy.stats.norm.pdf(
            np.expand_dims(trial_diff_obs, axis=1),
            loc=strength_means,
            scale=np.sqrt(2) * np.ones_like(strength_means))  # scale is std dev

        # P(o_{<=t} | \mu_n, s_n) = P(o_{<=t} | \mu_n)
        # shape = (num of observations, # of possible signed stimuli strengths)
        running_diff_obs_likelihood = np.cumprod(
            individual_diff_obs_likelihood,
            axis=0)

        # P(o_{<=t}, \mu_n, s_n | history) = P(o_{<=t} | \mu_n, s_n) P(\mu_n, s_n | history)
        # shape = (num of observations, # of possible signed stimuli strengths, # of trial sides i.e. 2)
        running_diff_obs_stim_side_strength_joint_prob = np.einsum(
            'ij,jk->ijk',
            running_diff_obs_likelihood,
            stim_side_strength_joint_prob)
        assert len(running_diff_obs_stim_side_strength_joint_prob.shape) == 3

        # marginalize out mu_n
        # shape = (num of observations, # of trial sides i.e. 2)
        running_diff_obs_stim_side_joint_prob = np.sum(
            running_diff_obs_stim_side_strength_joint_prob,
            axis=1)
        assert len(running_diff_obs_stim_side_joint_prob.shape) == 2

        # normalize by p(o_{<=t})
        # shape = (num of observations, # of trial sides i.e. 2)
        running_diff_obs_marginal_prob = np.sum(
            running_diff_obs_stim_side_joint_prob,
            axis=1)
        assert len(running_diff_obs_marginal_prob.shape) == 1

        # shape = (num of observations, # of trial sides i.e. 2)
        optimal_stim_posterior = np.divide(
            running_diff_obs_stim_side_joint_prob,
            np.expand_dims(running_diff_obs_marginal_prob, axis=1)  # expand to broadcast
        )
        assert np.allclose(
            np.sum(optimal_stim_posterior, axis=1),
            np.ones(len(optimal_stim_posterior)))

        session_data.loc[dt_indices, 'optimal_stimulus_posterior_right'] = \
            optimal_stim_posterior[:, 1]

    # determine ideal Bayesian observer action i.e. the MAP
    session_data['optimal_action_side'] = \
        2 * session_data['optimal_stimulus_posterior_right'].round() - 1
    session_data['optimal_correct_action_taken'] = \
        session_data['optimal_action_side'] == session_data['trial_side']
    # checking equality with one nan value (e.g. 1 == nan) returns False instead nan
    session_data.loc[pd.isna(session_data['optimal_action_side']),
                     'optimal_correct_action_taken'] = np.nan

    # log fraction of correct actions
    optimal_correct_action_taken_by_action_taken = session_data['optimal_correct_action_taken'].mean()
    logging.info(f'Optimal Fraction of Correct Actions Taken by Total Actions Taken: '
                 f'{optimal_correct_action_taken_by_action_taken}')

    # temp = session_data.loc[0:11, ['block_index', 'trial_index', 'rnn_step_index', 'signed_trial_strength', 'left_stimulus', 'right_stimulus', 'optimal_stimulus_prior_right', 'optimal_stimulus_posterior_right']]


def compute_optimal_prob_correct_blockless(session_data,
                                           rnn_steps_before_stimulus):
    # keep only last dts in trials
    trial_end_data = session_data[session_data.trial_end == 1.]
    possible_num_obs_within_trial = np.sort(trial_end_data['rnn_step_index'].unique())
    possible_num_obs_within_trial -= rnn_steps_before_stimulus - 1  # 0th RNN step is 1st observation

    optimal_prob_correct_after_num_obs_blockless_by_trial_strength = pd.DataFrame(
        np.nan,  # initialize all to nan
        index=possible_num_obs_within_trial,
        columns=trial_end_data.trial_strength.unique(),
        dtype=np.float16)
    for mu, number_of_trials in trial_end_data.groupby(['trial_strength']).size().iteritems():
        ideal_prob_correct_after_dt_per_trial_given_mu = np.array([
            1 - norm.cdf(-mu * np.sqrt(num_obs_within_trial)) if num_obs_within_trial > 0 else 0.5
            for num_obs_within_trial in possible_num_obs_within_trial])
        fraction_of_trials = number_of_trials / len(trial_end_data)
        optimal_prob_correct_after_num_obs_blockless_by_trial_strength[
            mu] = ideal_prob_correct_after_dt_per_trial_given_mu

    # reorder columns
    optimal_prob_correct_after_num_obs_blockless_by_trial_strength = \
        optimal_prob_correct_after_num_obs_blockless_by_trial_strength.reindex(
            sorted(optimal_prob_correct_after_num_obs_blockless_by_trial_strength.columns), axis=1)

    # average over possible trial strengths
    optimal_prob_correct_after_num_obs_blockless = optimal_prob_correct_after_num_obs_blockless_by_trial_strength.mean(
        axis=1)

    return optimal_prob_correct_after_num_obs_blockless, \
           optimal_prob_correct_after_num_obs_blockless_by_trial_strength


def compute_optimal_reward_rate_blockless(optimal_prob_correct_after_num_obs_blockless,
                                          optimal_prob_correct_after_num_obs_blockless_by_trial_strength,
                                          time_delay_penalty):
    penalty_after_num_obs = optimal_prob_correct_after_num_obs_blockless.index.values * time_delay_penalty

    optimal_reward_rate_after_num_obs_blockless = 2 * optimal_prob_correct_after_num_obs_blockless - 1
    optimal_reward_rate_after_num_obs_blockless = optimal_reward_rate_after_num_obs_blockless.subtract(
        penalty_after_num_obs, axis=0)

    optimal_reward_rate_after_num_obs_blockless_by_trial_strength = \
        2 * optimal_prob_correct_after_num_obs_blockless_by_trial_strength - 1

    optimal_reward_rate_after_num_obs_blockless_by_trial_strength = \
        optimal_reward_rate_after_num_obs_blockless_by_trial_strength.subtract(
            penalty_after_num_obs, axis=0)

    return optimal_reward_rate_after_num_obs_blockless, \
           optimal_reward_rate_after_num_obs_blockless_by_trial_strength


def compute_psytrack_fit(session_data):
    # need to add 1 because psytrack expects 1s & 2s, not 0s & 1s
    psytrack_model_choice = session_data['actions_chosen'].values[1:] + 1
    if np.var(psytrack_model_choice) < 0.025:
        logging.info('Model made only one action')
        return
    psytrack_stimuli = session_data['stimuli'].values[1:].reshape(-1, 1)
    psytrack_rewards = session_data['rewards'].values[:-1].reshape(-1, 1)

    # psytrack inputs need to be shaped (N, M), where N is number of trials and
    # M is arbitrary integer
    psytrack_inputs = dict(
        s1=psytrack_stimuli,
        s2=psytrack_rewards)

    psytrack_data = dict(
        y=psytrack_model_choice,
        inputs=psytrack_inputs,
        name='Temp')
    weights_dict = dict(
        # bias=1,
        s1=1,
        s2=1)  # only fit first column (there is only 1 column)
    total_num_weights = np.sum([weights_dict[i] for i in weights_dict.keys()])
    hyperparameters = dict(
        sigInit=np.power(2, 4),  # recommended
        sigma=[np.power(2., -4) for _ in range(total_num_weights)],  # recommended
        sigDay=None)  # recommended
    hyperparameters_to_optimize = ['sigma']

    # try:
    # TODO: figure out why this fails
    hyp, evd, wMAP, hess = hyperOpt(
        psytrack_data,
        hyperparameters,
        weights_dict,
        hyperparameters_to_optimize)

    # get uncertainty estimates
    credibleInt = getCredibleInterval(hess)

    psytrack_fit_output = dict(
        hyp=hyp,
        evd=evd,
        wMAP=wMAP,
        hess=hess,
        credibleInt=credibleInt)

    return psytrack_fit_output


def fit_reduced_dim_dynamics(session_data,
                             pca,
                             task_aligned_hidden_states):
    left_stimulus = np.expand_dims(
        session_data['left_stimulus'].values,
        axis=1)
    right_stimulus = np.expand_dims(
        session_data['right_stimulus'].values,
        axis=1)
    feedback = np.expand_dims(
        session_data['reward'].shift(1).fillna(0),
        axis=1)
    hidden_states = np.stack(session_data['hidden_state'].values.tolist()).squeeze(1)
    # add a row of zeros
    task_aligned_hidden_states = np.pad(
        task_aligned_hidden_states,
        pad_width=[(1, 0), (0, 0)])

    # normalize to ensure f() is invertible (in this case, tanh)
    # multiply by 1.01 to ensure abs(value) < 1
    max_task_aligned_hidden_states = 1.01 * np.max(np.abs(task_aligned_hidden_states))
    task_aligned_hidden_states /= max_task_aligned_hidden_states

    # project f^{-1}(h_n) = Ax + Bu to low dimension
    # y is target i.e. inverted PCA hidden states
    y = np.arctanh(task_aligned_hidden_states[1:])
    X = np.concatenate(
        (task_aligned_hidden_states[:-1], left_stimulus, right_stimulus, feedback),
        axis=1)

    # for condition_name, condition_idx in zip(condition_names, condition_indices):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=.33)
    lr = LinearRegression(fit_intercept=True, normalize=True)
    lr.fit(X=X_train, y=y_train)
    A_prime, B_prime = lr.coef_[:, :2], lr.coef_[:, 2:]
    r_squared = lr.score(X_test, y_test)
    logging.info(f'R^2: {r_squared}', )
    logging.info(f'A prime:\n{A_prime}', )
    logging.info(f'B prime:\n{B_prime}')

    task_aligned_model_states = np.zeros_like(task_aligned_hidden_states)
    task_aligned_model_state = np.zeros(2)
    inputs = X[:, 2:]
    for i in range(len(task_aligned_model_states) - 1):
        task_aligned_model_state = np.tanh(A_prime @ task_aligned_model_state + B_prime @ inputs[i, :])
        task_aligned_model_states[i + 1, :] = task_aligned_model_state

    control_model_states = np.zeros_like(task_aligned_hidden_states)
    A_rand = np.random.normal(size=A_prime.shape)
    B_rand = np.random.normal(size=B_prime.shape)
    control_model_state = np.zeros(2)
    inputs = X[:, 2:]
    for i in range(len(task_aligned_model_states) - 1):
        control_model_state = np.tanh(A_rand @ control_model_state + B_rand @ inputs[i, :])
        control_model_states[i + 1, :] = control_model_state

    task_aligned_hidden_states *= max_task_aligned_hidden_states
    task_aligned_model_states *= max_task_aligned_hidden_states
    model_states = pca.inverse_transform(task_aligned_model_states)

    trajectory_names = ['hidden_states', 'task_aligned_hidden_states', 'task_aligned_model_states',
                        'model_states', 'control']
    trajectories = [hidden_states, task_aligned_hidden_states, task_aligned_model_states,
                    model_states, control_model_states]
    max_delta = 20
    error_accumulation_df = pd.DataFrame(
        np.nan,
        columns=['name', 'delta', 'norm_mean', 'norm_var'],
        index=np.arange(len(trajectories) * max_delta))
    error_accumulation_df.name = error_accumulation_df.name.astype(np.object)
    i = 0
    for trajectory_name, trajectory in zip(trajectory_names, trajectories):
        for delta in range(max_delta):
            # numpy diff strangely return array if n=0, instead of all zeros
            if delta == 0:
                diffs = np.zeros_like(trajectory)
            else:
                diffs = np.diff(trajectory, n=delta, axis=0)
            diffs_norms = np.linalg.norm(diffs, axis=1)
            norm_mean = np.mean(diffs_norms) / np.sqrt(trajectory.shape[1])
            norm_var = np.var(diffs_norms) / np.sqrt(trajectory.shape[1])

            error_accumulation_df.at[i, 'name'] = trajectory_name
            error_accumulation_df.at[i, 'delta'] = delta
            error_accumulation_df.at[i, 'norm_mean'] = norm_mean
            error_accumulation_df.at[i, 'norm_var'] = norm_var
            i += 1

    reduced_dynamics_results = dict(
        A_prime=A_prime,
        B_prime=B_prime,
        error_accumulation_df=error_accumulation_df)

    return reduced_dynamics_results


def load_mice_behavioral_data(mouse_behavior_dir_path):
    mice_behavior_df = []
    for mouse_csv_path in os.listdir(mouse_behavior_dir_path):
        if mouse_csv_path.split('_')[-1].startswith('endtrain'):
            continue
        mouse_behav_df = pd.read_csv(os.path.join(mouse_behavior_dir_path, mouse_csv_path))
        mice_behavior_df.append(mouse_behav_df)
    mice_behavior_df = pd.concat(mice_behavior_df)

    # drop unbiased blocks
    mice_behavior_df = mice_behavior_df[mice_behavior_df['stim_probability_left'] != .5]

    # convert stim_prob_left from 0.8 and 0.2 to -1, 1
    mice_behavior_df['block_side'] = np.where(
        mice_behavior_df['stim_probability_left'] == .8, -1, 1)

    # convert postion from +-35.0 to -1, 1
    mice_behavior_df['stimulus_side'] = mice_behavior_df['position'] / 35.

    # determine whether stimulus side and block side are concordant
    mice_behavior_df['concordant_trial'] = mice_behavior_df['block_side'] == mice_behavior_df['stimulus_side']

    # create signed contrast
    mice_behavior_df['signed_contrast'] = mice_behavior_df['contrast'] * mice_behavior_df['stimulus_side']

    mice_behavior_data_results = dict(
        mice_behavior_df=mice_behavior_df)

    return mice_behavior_data_results


def run_model_one_step(model,
                       stimulus,
                       feedback,
                       hidden_states=None):
    """

    :param model:
    :param stimulus: shape (batch size, num steps, stimulus dimension)
    :param feedback: shape (batch size, num_steps, )
    :param hidden_states: shape (batch_size, hidden layers, hidden state size)
    :param input_requires_grad:
    :return:
    """

    feedback = feedback.double()
    stimulus = stimulus.double()

    # set model's hidden states to given hidden states
    if hidden_states is None:
        model.core_hidden = None
    else:
        hidden_states = hidden_states.double()

        if model.model_str in {'rnn', 'gru'}:
            # Pytorch expects shape (# hidden layers, batch size, hidden state size)
            model.core_hidden = hidden_states.permute(1, 0, 2)
        elif model.model_str in {'lstm'}:
            # PyTorch expects a 2-tuple of shape (# hidden layers, batch size, hidden state size)
            model.core_hidden = (hidden_states[:, :, :, 0].permute(1, 0, 2),
                                 hidden_states[:, :, :, 1].permute(1, 0, 2),)

    # create single step model input
    model_input = dict(
        stimulus=stimulus,
        reward=feedback)

    model_forward_output = model(model_input)

    return model_forward_output


def sample_model_states_in_state_space(projection_obj,
                                       xrange,
                                       yrange,
                                       pca_hidden_states):
    # projection_obj should be either pca, jlm, etc.
    # TODO: generalize to jlm

    # compute convex hull encompassing network activity
    convex_hull = scipy.spatial.Delaunay(pca_hidden_states)

    # sample possible activity uniformly over the plane, then exclude points
    # outside the convex hull
    pc1_values, pc2_values = np.meshgrid(
        np.linspace(xrange[0], xrange[1], num=50),
        np.linspace(yrange[0], yrange[1], num=50))
    pca_hidden_states = np.stack((pc1_values.flatten(), pc2_values.flatten())).T
    in_hull_indices = test_points_in_hull(p=pca_hidden_states, hull=convex_hull)
    pca_hidden_states = pca_hidden_states[in_hull_indices]
    logging.info(f'Number of sampled states: {len(pca_hidden_states)}')
    sampled_states = projection_obj.inverse_transform(pca_hidden_states)
    return sampled_states


def test_points_in_hull(p, hull):
    """
    Test if points `p` are in `hull`. Stolen from
    https://stackoverflow.com/a/16898636/4570472.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0
