import community
from itertools import product
import pandas as pd
import networkx as nx
import networkx.algorithms.community
import networkx.drawing
import numpy as np
from psytrack.helper.invBlkTriDiag import getCredibleInterval
from psytrack.hyperOpt import hyperOpt
import re
from scipy.linalg import solve_discrete_lyapunov
import scipy.spatial
from sklearn.decomposition.pca import PCA
from sklearn.random_projection import GaussianRandomProjection
import statsmodels.api as sm
import torch
import torch.autograd
import torch.optim

from utils.env import create_custom_worlds
from utils.run import run_envs

possible_stimuli = torch.DoubleTensor(
    [[2.2, 0.2],
     [0.2, 0.2],
     [0.2, 2.2],
     [0., 0.],
     [0., 0.],
     [0., 0.]])

possible_feedback = torch.DoubleTensor(
    [[0],
     [0],
     [0],
     [-1],
     [1],
     [0]])


def add_analysis_data_to_hook_input(hook_input):
    # convert from shape (number of total time steps, num hidden layers, hidden size) to
    # shape (number of total time steps, num hidden layers * hidden size)
    reshaped_hidden_states = hook_input['hidden_states'].reshape(
        hook_input['hidden_states'].shape[0], -1)

    hidden_states_pca_results = compute_model_hidden_states_pca(
        hidden_states=reshaped_hidden_states)

    hidden_states_jl_results = compute_model_hidden_states_johnson_lindenstrauss(
        hidden_states=reshaped_hidden_states)

    model_readout_vectors_results = compute_model_readout_vectors(
        session_data=hook_input['session_data'],
        hidden_states=reshaped_hidden_states,
        model_readout_weights=hook_input['model'].readout.weight.data.numpy(),
        pca_hidden_states=hidden_states_pca_results['pca_hidden_states'],
        pca=hidden_states_pca_results['pca'])

    fixed_points_results = compute_model_fixed_points_by_stimulus_and_feedback(
        model=hook_input['model'],
        pca=hidden_states_pca_results['pca'],
        pca_xrange=hidden_states_pca_results['pca_xrange'],
        pca_yrange=hidden_states_pca_results['pca_yrange'],
        jlm=hidden_states_jl_results['jlm'],
        jlm_xrange=hidden_states_jl_results['jl_xrange'],
        jlm_yrange=hidden_states_jl_results['jl_yrange'],
        pca_sampled_states=hidden_states_pca_results['pca_hidden_states'],
        trial_readout_vector=model_readout_vectors_results['trial_readout_vector'],
        block_readout_vector=model_readout_vectors_results['block_readout_vector'],
        num_grad_steps=500)

    eigenvalues_svd_results = compute_eigenvalues_svd(
        matrix=reshaped_hidden_states)

    # add results to hook_input
    result_dicts = [
        hidden_states_jl_results,
        hidden_states_pca_results,
        model_readout_vectors_results,
        fixed_points_results,
        eigenvalues_svd_results]
    for result_dict in result_dicts:
        hook_input.update(result_dict)


def compute_eigenvalues_svd(matrix):
    """
    matrix should have shape (num_samples, num_features)
    """
    feature_means = np.mean(matrix, axis=0)
    s = np.linalg.svd(matrix - feature_means, full_matrices=False, compute_uv=False)

    # eigenvalues
    variance_explained = np.power(s, 2) / (matrix.shape[0] - 1)
    frac_variance_explained = np.cumsum(variance_explained / np.sum(variance_explained))

    eigenvalues_svd_results = dict(
        variance_explained=variance_explained,
        frac_variance_explained=frac_variance_explained
    )

    return eigenvalues_svd_results


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
            second_states = final_states.clone()
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

    print('Stimulus val: ', stimulus_val.numpy())
    print('Feedback val: ', feedback_val.numpy())
    normalized_displacement_norm = torch.div(
        displacement_norm,
        torch.norm(final_states, dim=(1, 2)))
    print('Minimum displacement norm: ', torch.min(displacement_norm).item())

    model_fixed_points_results = dict(
        second_states=second_states.detach().numpy(),
        final_states=final_states.detach().numpy(),
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


def compute_model_fixed_points_basins_of_attraction(fixed_point_df,
                                                    model):
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
        'energy']
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
        'left_stimulus', 'right_stimulus', 'feedback'])):

        fixed_points_basins_df.at[i, 'left_stimulus'] = lstim
        fixed_points_basins_df.at[i, 'right_stimulus'] = rstim
        fixed_points_basins_df.at[i, 'feedback'] = fdbk

        # only original jacobian is Hurwitz. Maybe consider PCA jacobian later.
        stable_fp_df = fixed_point_subset_df[
            (fixed_point_subset_df['jacobian_hidden_stable'] == 1) &
            (fixed_point_subset_df['jacobian_pca_stable'] == 1)]

        if len(stable_fp_df) == 0:
            continue

        # filter for MOST fixed point
        minimum_displacement = fixed_point_subset_df['displacement_norm'].min()
        if minimum_displacement > 0.01:
            continue
        minimum_displacement_index = fixed_point_subset_df['displacement_norm'].idxmin()
        fixed_point_state = fixed_point_subset_df.at[
            minimum_displacement_index, 'final_sampled_state']
        pca_fixed_point_state = fixed_point_subset_df.at[
            minimum_displacement_index, 'final_pca_sampled_state']

        Q = np.eye(len(fixed_point_state.flatten()))
        lambda_min_Q = np.min(np.linalg.eigvals(Q))
        A = np.array(fixed_point_subset_df.at[minimum_displacement_index, 'jacobian_hidden'])
        lambda_max_ATA = np.max(np.linalg.eigvals(A.T @ A))
        P = solve_discrete_lyapunov(a=A, q=Q)
        np.testing.assert_almost_equal(P, P.T)
        lambda_max_P = np.max(np.linalg.eigvals(P))

        # find parameter controlling basin of attraction
        gamma = np.max(np.roots([
            -lambda_min_Q,
            2 * lambda_max_P * np.sqrt(lambda_max_ATA),
            lambda_max_P]))

        # shrink gamma a little to exclude the root
        gamma -= .01

        if gamma <= 0:
            print('No positive roots found for gamma')
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

        # Let the system be x(t+1) = f(x(t)). Following the coordinate transform,
        # Taylor series about the fixed point x^* = 0. Then f(x) = f(x^*) + A x + g(x),
        # where A is the Jacobian evaluated at the fixed point.
        ts_linearized_component = np.einsum(
            'ijl,kl->ijk',
            coord_transform_initial_states,
            A)

        # g(x) is the Taylor Series remainder
        # TODO: because f(x^*) is not exactly zero, subtract it i.e. g(x) = f(x) - f(x^*) - A x
        ts_remainder = coord_transform_second_states - ts_linearized_component

        # select points such that ||g(x)|| < \gamma ||Ax||
        g_x_norm = np.linalg.norm(ts_remainder, axis=(1, 2))
        x_norm = np.linalg.norm(coord_transform_initial_states, axis=(1, 2))
        within_region_indices = g_x_norm < (gamma * x_norm)
        if not np.any(within_region_indices):
            continue
        hidden_states_in_basin = initial_sampled_states[within_region_indices]
        initial_pca_states_in_basin = np.stack(fixed_point_subset_df.loc[
            within_region_indices, 'initial_pca_sampled_state'].values.tolist())
        within_region_coord_transform_states = coord_transform_initial_states[
            within_region_indices].squeeze(1)
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
                                                        pca_sampled_states,
                                                        trial_readout_vector,
                                                        block_readout_vector,
                                                        num_grad_steps=100):
    assert num_grad_steps > 2

    sampled_states = sample_model_states_in_state_space(
        projection_obj=pca,
        xrange=pca_xrange,
        yrange=pca_yrange,
        projected_sampled_states=pca_sampled_states)
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
        'final_sampled_state',
        'final_pca_sampled_state',
        'displacement',
        'displacement_norm',
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
                   'final_sampled_state', 'final_pca_sampled_state', 'jacobian_hidden',
                   'jacobian_hidden_sym', 'jacobian_hidden_eigenspectrum',
                   'displacement']:
        fixed_point_df[column] = fixed_point_df[column].astype(object)

    print(f'Computing fixed points using {num_grad_steps} gradient steps')
    for row_group, (feedback_val, stimulus_val) in enumerate(zip(possible_feedback, possible_stimuli)):

        model_fixed_points_results = compute_model_fixed_points(
            model=model,
            initial_states=initial_sampled_states,
            stimulus_val=stimulus_val,
            feedback_val=feedback_val,
            num_grad_steps=num_grad_steps)
        stimulus = model_fixed_points_results['stimulus']
        feedback = model_fixed_points_results['feedback']
        final_sampled_states = model_fixed_points_results['final_states']
        final_pca_sampled_states = pca.transform(
            final_sampled_states.reshape(len(final_sampled_states), -1))

        fixed_point_subrows = {
            'subjective_block_side': subjective_block_side[:, 0],
            'subjective_trial_side': subjective_trial_side[:, 0],
            'left_stimulus': stimulus[:, 0, 0],
            'right_stimulus': stimulus[:, 0, 1],
            'feedback': feedback[:, 0],
            'initial_sampled_state': initial_sampled_states.numpy(),
            'initial_pca_sampled_state': pca_sampled_states,
            'second_sampled_state': model_fixed_points_results['second_states'],
            'final_sampled_state': final_sampled_states,
            'final_pca_sampled_state': final_pca_sampled_states,
            'displacement': model_fixed_points_results['displacement'],
            'displacement_norm': model_fixed_points_results['displacement_norm'],
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
        model=model,
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
        displacement_norm < np.quantile(displacement_norm, .01)).astype(np.float16)

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
        displacement_norm < 0.01).astype(np.float16)

    return model_jacobians_results


def compute_model_fixed_points_jacobians_projected(fixed_point_df,
                                                   pca,
                                                   jlm):

    # necessary because Pandas object series can't be directly convert to array
    jacobians = np.stack(fixed_point_df['jacobian_hidden'].values.tolist())
    jacobians_stable = fixed_point_df['jacobian_hidden_stable']
    print('Fraction of stable Jacobians: ',
          np.mean(jacobians_stable))

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
        pca_displacement_norm < np.quantile(pca_displacement_norm, .01),
    ).astype(np.float16)
    print('Fraction of stable PCA Jacobians: ', np.mean(jacobians_pca_stable))

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
        jlm_displacement_norm < np.quantile(jlm_displacement_norm, .01),
    ).astype(np.float16)
    print('Fraction of stable JL Jacobians: ', np.mean(jacobians_jlm_stable))

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


def compute_model_hidden_state_vector_field(model,
                                            session_data,
                                            hidden_states,
                                            pca,
                                            pca_hidden_states):
    columns = [
        'session_data_indices',
        'rnn_step_within_session',
        'block_index',
        'trial_index',
        'rnn_step_index',
        'block_side',
        'trial_side',
        'trial_strength',
        'left_stimulus',
        'right_stimulus',
        'reward',
        'correct_action_prob',
        'left_action_prob',
        'right_action_prob',
        'hidden_state',
        'pca_hidden_state',
        'next_hidden_state',
        'pca_next_hidden_state',
        'hidden_state_difference']

    vector_fields_df = pd.DataFrame(
        columns=columns,
        dtype=np.float16)

    # enable storing hidden states in the dataframe.
    # need to make the column have type object to doing so possible
    for column in ['hidden_state', 'pca_hidden_state', 'next_hidden_state',
                   'pca_next_hidden_state', 'hidden_state_difference']:
        vector_fields_df[column] = vector_fields_df[column].astype(object)

    # sample subset of indices
    random_subset_indices = np.random.choice(
        session_data.index,
        replace=False,
        size=min(1000, len(session_data)))

    sampled_hidden_states = hidden_states[random_subset_indices]
    sampled_pca_hidden_states = pca_hidden_states[random_subset_indices]

    for feedback, stimulus in product(possible_feedback, possible_stimuli):
        # shape: (len(random subset indices, 1)
        feedback = torch.stack(len(random_subset_indices) * [feedback], dim=0)

        # shape: (len(random subset indices), 1, 2)
        stimulus = torch.stack(len(random_subset_indices) * [stimulus], dim=0).unsqueeze(1)

        model_forward_output = run_model_one_step(
            model=model,
            stimulus=stimulus,
            feedback=feedback,
            hidden_states=torch.from_numpy(sampled_hidden_states))

        next_sampled_hidden_states = model_forward_output['core_hidden'].detach().numpy()

        next_sampled_pca_hidden_states = pca.transform(
            next_sampled_hidden_states.reshape(len(random_subset_indices), -1))

        displacement_vector = next_sampled_pca_hidden_states - sampled_pca_hidden_states

        vector_fields_subrows_df = dict(
            displacement_vector=displacement_vector,
            random_subset_indices=random_subset_indices,
            sampled_hidden_states=sampled_hidden_states,
            sampled_pca_hidden_states=sampled_pca_hidden_states,
            next_sampled_hidden_states=next_sampled_hidden_states,
            next_sampled_pca_hidden_states=next_sampled_pca_hidden_states)

        vector_fields_subrows_df = pd.DataFrame(vector_fields_subrows_df)

        vector_fields_df = pd.concat((vector_fields_df, vector_fields_subrows_df))

    return vector_fields_df


def compute_model_hidden_states_johnson_lindenstrauss(hidden_states):
    # ensure hidden states have 2 dimensions
    assert len(hidden_states.shape) == 2
    jlmatrix = GaussianRandomProjection(n_components=2)
    jlmatrix.fit(hidden_states)
    jl_hidden_states = jlmatrix.fit_transform(hidden_states)

    min_x, max_x = min(jl_hidden_states[:, 0]), max(jl_hidden_states[:, 0])
    min_y, max_y = min(jl_hidden_states[:, 1]), max(jl_hidden_states[:, 1])

    hidden_states_pca_results = dict(
        jl_hidden_states=jl_hidden_states,
        jl_xrange=(min_x, max_x),
        jl_yrange=(min_y, max_y),
        jlm=jlmatrix)

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

        # because feedback has dim 2 instead of dim 3, the jacobian will be shape (hidden dim, )
        # add an extra dimension for consistency with other two Jacobians
        if name == 'feedback':
            jacobian = torch.unsqueeze(jacobian, dim=2)

        model_jacobians_results['jacobian_' + name] = jacobian.detach().numpy()

    # ensures no problems arise later
    for input_var in input_vars:
        input_var.requires_grad_(False)

    return model_jacobians_results


def compute_model_hidden_states_pca(hidden_states):
    # ensure hidden states have 2 dimensions
    assert len(hidden_states.shape) == 2
    pca = PCA(n_components=2)
    pca.fit(hidden_states)
    pca_hidden_states = pca.transform(hidden_states)

    min_x, max_x = min(pca_hidden_states[:, 0]), max(pca_hidden_states[:, 0])
    min_y, max_y = min(pca_hidden_states[:, 1]), max(pca_hidden_states[:, 1])

    hidden_states_pca_results = dict(
        pca_hidden_states=pca_hidden_states,
        pca_xrange=(min_x, max_x),
        pca_yrange=(min_y, max_y),
        pca=pca)

    return hidden_states_pca_results


def compute_model_readout_vectors(session_data,
                                  model_readout_weights,
                                  hidden_states,
                                  pca_hidden_states,
                                  pca):
    block_sides = session_data.block_side.values
    trial_sides = session_data.trial_side.values

    # select RIGHT trial side readout vector
    trial_readout_vector = model_readout_weights[-1, np.newaxis, :]
    trial_readout_vector /= np.linalg.norm(trial_readout_vector)
    pca_trial_readout_vector = pca.transform(trial_readout_vector)[0]

    # ensure PCs match right trial side; otherwise, reflect PCs
    mean_right_trial_hidden_state_projection_onto_right_readout_weight = np.dot(
        pca_hidden_states[trial_sides == 1],
        pca_trial_readout_vector).mean()
    if mean_right_trial_hidden_state_projection_onto_right_readout_weight < 0:
        print('Swapped readout vector direction')
        pca.components_ *= -1.
        pca_trial_readout_vector *= -1

    # select RIGHT block side readout vector
    logistic_regression = sm.Logit(
        endog=(1 + block_sides) / 2,  # transform from {-1, 1} to {0, 1}
        exog=hidden_states)
    logistic_regression_result = logistic_regression.fit()
    session_data['classifier_block_side'] = 2. * logistic_regression_result.predict(hidden_states) - 1.
    block_readout_vector = np.expand_dims(logistic_regression_result.params, 0)
    block_readout_vector /= np.linalg.norm(block_readout_vector)
    pca_block_readout_vector = pca.transform(block_readout_vector)[0]

    radians_btwn_pca_trial_block_vectors = np.arccos(np.dot(
        pca_block_readout_vector / np.linalg.norm(pca_block_readout_vector),
        pca_trial_readout_vector / np.linalg.norm(pca_trial_readout_vector)))

    degrees_btwn_pca_trial_block_vectors = round(
        180 * radians_btwn_pca_trial_block_vectors / np.pi)

    print(f'Degrees between vectors: {degrees_btwn_pca_trial_block_vectors}')

    block_readout_weights_results = dict(
        trial_readout_vector=trial_readout_vector,
        pca_trial_readout_vector=pca_trial_readout_vector,
        block_readout_vector=block_readout_vector,
        pca_block_readout_vector=pca_block_readout_vector,
        radians_btwn_pca_trial_block_vectors=radians_btwn_pca_trial_block_vectors,
        degrees_btwn_pca_trial_block_vectors=degrees_btwn_pca_trial_block_vectors)

    return block_readout_weights_results


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

    import matplotlib.pyplot as plt
    plt.show()

    print(10)


def compute_psytrack_fit(session_data):
    # need to add 1 because psytrack expects 1s & 2s, not 0s & 1s
    psytrack_model_choice = session_data['actions_chosen'].values[1:] + 1
    if np.var(psytrack_model_choice) < 0.025:
        print('Model made only one action')
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


def nonlinear_control(hook_input):
    import matplotlib.pyplot as plt

    print(10)


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
                                       projected_sampled_states):
    # projection_obj should be either pca, jlm, etc.
    # TODO: generalize to jlm

    # compute convex hull encompassing network activity
    convex_hull = scipy.spatial.Delaunay(projected_sampled_states)

    # sample possible activity uniformly over the plane, then exclude points
    # outside the convex hull
    pc1_values, pc2_values = np.meshgrid(
        np.linspace(xrange[0], xrange[1], num=50),
        np.linspace(yrange[0], yrange[1], num=50))
    projected_sampled_states = np.stack((pc1_values.flatten(), pc2_values.flatten())).T
    in_hull_indices = test_points_in_hull(p=projected_sampled_states, hull=convex_hull)
    projected_sampled_states = projected_sampled_states[in_hull_indices]
    print('Number of sampled states: ', len(projected_sampled_states))
    sampled_states = projection_obj.inverse_transform(projected_sampled_states)
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
