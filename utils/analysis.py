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
from sklearn.decomposition.pca import PCA
import statsmodels.api as sm
import torch
import torch.autograd
import torch.optim

from utils.env import create_custom_worlds
from utils.run import run_envs


possible_stimuli = torch.DoubleTensor(
    [[2.2, 0.2],
     [0.2, 0.2],
     [0.2, 2.2]])

possible_feedback = torch.DoubleTensor(
    [[-1],
     [0],
     [1]])


def add_analysis_data_to_hook_input(hook_input):

    # convert from shape (number of total time steps, num hidden layers, hidden size) to
    # shape (number of total time steps, num hidden layers * hidden size)
    reshaped_hidden_states = hook_input['hidden_states'].reshape(
        hook_input['hidden_states'].shape[0], -1)

    hidden_states_pca_results = compute_model_hidden_states_pca(
            hidden_states=reshaped_hidden_states)

    model_readout_vectors_results = compute_model_readout_vectors(
        session_data=hook_input['session_data'],
        hidden_states=reshaped_hidden_states,
        model_readout_weights=hook_input['model'].readout.weight.data.numpy(),
        pca_hidden_states=hidden_states_pca_results['pca_hidden_states'],
        pca=hidden_states_pca_results['pca'])

    # fixed_points_by_side_by_stimuli = compute_model_fixed_points(
    #     model=model,
    #     pca=pca,
    #     pca_hidden_states=pca_hidden_states,
    #     session_data=run_envs_output['session_data'],
    #     hidden_states=hidden_states,
    #     num_grad_steps=50)

    eigenvalues_svd_results = compute_eigenvalues_svd(
        matrix=reshaped_hidden_states)

    # add results to hook_input
    result_dicts = [hidden_states_pca_results,
                    model_readout_vectors_results,
                    eigenvalues_svd_results]
    for result_dict in result_dicts:
        # TODO: check that update is correct
        hook_input.update(result_dict)
        # for key, value in result_dict.items():
        #     hook_input[key] = value


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


def compute_jacobians_by_side_by_stimuli(model,
                                         trial_data,
                                         fixed_points_by_side_by_stimuli):

    num_layers = model.model_kwargs['core_kwargs']['num_layers']
    hidden_size = model.model_kwargs['core_kwargs']['hidden_size']
    input_size = model.input_size
    num_basis_vectors = num_layers * hidden_size

    unit_basis_vectors = torch.eye(n=num_basis_vectors).reshape(num_basis_vectors, num_layers, hidden_size)

    # we need to repeat each unit basis vector, once for each hidden dimension
    # unit_basis_vectors = torch.cat([unit_basis_vectors for _ in range(num_basis_vectors)])

    # consider only the "most" fixed point per stimulus/side
    num_fixed_points_to_consider = 1

    jacobians_by_side_by_stimuli = {}
    for side, fixed_points_by_stimuli_dict in fixed_points_by_side_by_stimuli.items():
        jacobians_by_side_by_stimuli[side] = {}
        for stimulus, fixed_points_dict in fixed_points_by_stimuli_dict.items():
            jacobians_by_side_by_stimuli[side][stimulus] = {}
            rewards = torch.from_numpy(
                trial_data.iloc[:num_fixed_points_to_consider]['reward'].to_numpy()).reshape(-1, 1)
            rewards = rewards.repeat(unit_basis_vectors.shape[0], 1).requires_grad_(True)
            _, left_stimulus, _, right_stimulus = re.split(r'[,=]]*', stimulus)
            left_stimulus, right_stimulus = float(left_stimulus), float(right_stimulus)

            stimuli = torch.zeros(
                size=(num_fixed_points_to_consider * unit_basis_vectors.shape[0], 1, 1)).fill_(stimulus)
            stimuli = stimuli.requires_grad_(True)

            hidden_states = fixed_points_dict['final_sampled_hidden_states'][:num_fixed_points_to_consider]
            hidden_states = hidden_states.repeat(unit_basis_vectors.shape[0], 1, 1)

            assert rewards.shape[0] == stimuli.shape[0] == hidden_states.shape[0]

            model_forward_output = run_model_one_step(
                model=model,
                stimulus=stimuli,
                feedback=rewards,
                hidden_states=hidden_states)

            for str_inputs in ['hidden', 'stimuli', 'rewards']:

                if str_inputs == 'hidden':
                    inputs = hidden_states
                    reshape_size = (num_basis_vectors, num_layers * hidden_size)
                elif str_inputs == 'stimuli':
                    inputs = stimuli
                    reshape_size = (num_basis_vectors, 1)
                elif str_inputs == 'rewards':
                    inputs = rewards
                    reshape_size = (num_basis_vectors, 1)

                # jacobian will have shape (num basis vectors, num hidden layers, hidden size)
                jacobian = torch.autograd.grad(
                    outputs=model_forward_output['core_output'],
                    inputs=inputs,
                    grad_outputs=unit_basis_vectors,
                    retain_graph=True,  # need to retain for next jacobian
                    only_inputs=True)[0]
                jacobian = jacobian.reshape(*reshape_size).numpy()

                jacobians_by_side_by_stimuli[side][stimulus][
                    f'{str_inputs}_to_hidden'] = jacobian

    return jacobians_by_side_by_stimuli


def compute_projected_hidden_state_trajectory_controlled(model,
                                                         pca):

    envs = create_custom_worlds(
        num_envs=1,
        blocks_per_session=12,
        tensorboard_writer=None)
    run_envs_output = run_envs(
        model=model,
        envs=envs)
    hidden_states = run_envs_output['hidden_states']
    projected_hidden_states = pca.transform(hidden_states.reshape(hidden_states.shape[0], -1))

    trajectory_controlled_output = dict(
        session_data=run_envs_output['session_data'],
        hidden_states=hidden_states,
        projected_hidden_states=projected_hidden_states,
    )

    return trajectory_controlled_output


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

    # ensure PCs match right block side; otherwise, reflect PCs
    mean_right_block_hidden_state_projection_onto_right_readout_weight = np.dot(
        pca_hidden_states[session_data.block_side == 1],
        pca_block_readout_vector).mean()
    if mean_right_block_hidden_state_projection_onto_right_readout_weight < 0:
        raise ValueError('Block side, trial side PCs disagree')

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
    total_weight_matrix[:input_num_units, input_num_units:input_num_units+recurrent_num_units] = \
        weights['input'].T
    total_weight_matrix[input_num_units:input_num_units+recurrent_num_units, input_num_units:input_num_units+recurrent_num_units] = \
        weights['recurrent'].T
    total_weight_matrix[input_num_units:input_num_units+recurrent_num_units, input_num_units+recurrent_num_units:] = \
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


def compute_model_fixed_points(model,
                               pca,
                               pca_hidden_states,
                               session_data,
                               hidden_states,
                               num_grad_steps=100):

    assert num_grad_steps > 0

    # project all hidden states using pca
    # reshape to (num trials, num layers * hidden dimension)

    # identify non-first block indices
    fixed_points_by_side_by_stimuli = {}
    for side, session_data_block_side in session_data.groupby('block_side'):
        fixed_points_by_side_by_stimuli[side] = dict()

        for possible_stimulus in possible_stimuli:

            initial_sampled_hidden_states = torch.from_numpy(hidden_states)
            pca_initial_sampled_hidden_states = pca_hidden_states

            # require grad to use fixed point finder i.e. minimize ||h_t - RNN(h_t, possible_stimulus)||
            final_sampled_hidden_states = initial_sampled_hidden_states.clone().requires_grad_(True)

            rewards = torch.from_numpy(session_data.reward.to_numpy()).reshape(-1, 1)

            # shape: (len(session_data_, 1, 2)
            stimuli = torch.stack(len(session_data) * [possible_stimulus], dim=0).unsqueeze(1)

            optimizer = torch.optim.SGD([final_sampled_hidden_states], lr=0.01)

            for _ in range(num_grad_steps):
                optimizer.zero_grad()
                model_forward_output = run_model_one_step(
                    model=model,
                    stimulus=stimuli,
                    feedback=rewards,
                    hidden_states=final_sampled_hidden_states)
                # non-LSTM shape: (session size, 1 time step, hidden state size)
                # LSTM shape: (session size, 1 time step, hidden state size, 2)
                model_forward_hidden_state = model_forward_output['core_hidden']
                displacement_vector = model_forward_hidden_state - final_sampled_hidden_states
                # if LSTM, merge last two dimension
                if len(displacement_vector.shape) == 4:
                    displacement_vector = displacement_vector.reshape(
                        (len(displacement_vector), 1, -1))
                    model_forward_hidden_state = model_forward_hidden_state.reshape(
                        (len(model_forward_hidden_state), 1, -1))
                displacement_vector_norm = torch.norm(
                    displacement_vector,
                    dim=(1, 2))
                normalized_displacement_vector_norm = torch.div(
                    displacement_vector_norm,
                    torch.norm(model_forward_hidden_state, dim=(1, 2)))
                # print(f'Norm of smallest displacement vector: {torch.min(normalized_displacement_vector_norm)}')
                loss = torch.mean(displacement_vector_norm)
                # print(f'Fixed point finder loss: {loss.item()}')
                loss.backward()
                optimizer.step()

            pca_final_sampled_hidden_states = pca.transform(
                final_sampled_hidden_states.reshape(len(session_data), -1).detach().numpy())
            pca_displacement_vector = pca.transform(
                displacement_vector.reshape(len(session_data), -1).detach().numpy())

            key = 'left={}, right={}'.format(
                possible_stimulus[0].item(),
                possible_stimulus[1].item())

            fixed_points_by_side_by_stimuli[side][key] = dict(
                displacement_vector=displacement_vector.detach().numpy(),
                displacement_vector_norm=displacement_vector_norm.detach().numpy(),
                normalized_displacement_vector_norm=normalized_displacement_vector_norm.detach().numpy(),
                pca_displacement_vector=pca_displacement_vector,
                initial_sampled_hidden_states=initial_sampled_hidden_states,
                pca_initial_sampled_hidden_states=pca_initial_sampled_hidden_states,
                final_sampled_hidden_states=final_sampled_hidden_states,
                pca_final_sampled_hidden_states=pca_final_sampled_hidden_states,
                num_grad_steps=num_grad_steps)

    return fixed_points_by_side_by_stimuli


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


def run_model_one_step(model,
                       stimulus,
                       feedback,
                       hidden_states=None):

    """

    :param model:
    :param stimulus: shape (batch size, num steps, stimulus dimension)
    :param feedback: shape (batch size, num_steps, )
    :param hidden_states:
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