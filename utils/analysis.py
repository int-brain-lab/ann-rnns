import numpy as np
from psytrack.helper.invBlkTriDiag import getCredibleInterval
from psytrack.hyperOpt import hyperOpt
from sklearn.decomposition.pca import PCA
import torch
import torch.autograd
import torch.optim


# def extract_eigenvalues_pca(matrix):
#     """
#     matrix should have shape (num_samples, num_features)
#     """
#     # TODO: don't use this!
#     pca = PCA()
#     pca.fit(matrix)
#     eigenvalues = pca.explained_variance_
#     return eigenvalues


def compute_eigenvalues_svd(matrix):
    """
    matrix should have shape (num_samples, num_features)
    """
    feature_means = np.mean(matrix, axis=0)
    s = np.linalg.svd(matrix - feature_means, full_matrices=False, compute_uv=False)
    eigenvalues = np.power(s, 2) / (matrix.shape[0] - 1)
    return eigenvalues


def compute_hidden_state_vector_field(model,
                                      trial_data,
                                      hidden_states):

    # project all hidden states using pca
    # reshape to (num trials, num layers * hidden dimension)
    projected_hidden_states, xrange, yrange, pca = compute_projected_hidden_states_pca(
        hidden_states=hidden_states.reshape(hidden_states.shape[0], -1))

    # identify non-first block indices
    possible_stimuli = np.linspace(-1.5, 1.5, 3)

    vector_fields_by_side_by_stimuli = {}
    for side, trial_data_preferred_side in trial_data.groupby('stimuli_preferred_sides'):
        vector_fields_by_side_by_stimuli[side] = dict()

        for possible_stimulus in possible_stimuli:

            non_first_block_indices = trial_data_preferred_side.index[
                trial_data_preferred_side['stimuli_block_number'] != 1].to_numpy()

            # sample subset of indices
            random_subset_indices = np.random.choice(
                non_first_block_indices,
                replace=False,
                size=150)

            sampled_hidden_states = hidden_states[random_subset_indices]
            projected_sampled_hidden_states = projected_hidden_states[random_subset_indices]

            rewards = torch.from_numpy(
                trial_data.iloc[random_subset_indices]['rewards'].to_numpy()).reshape(-1, 1)

            stimuli = torch.zeros(
                size=(len(random_subset_indices), 1, 1)).fill_(possible_stimulus)

            model_forward_output = run_model_one_step(
                model=model,
                stimuli=stimuli,
                rewards=rewards,
                hidden_states=torch.from_numpy(sampled_hidden_states))

            sampled_next_hidden_states = model_forward_output['core_hidden'].detach().numpy()

            projected_sampled_next_hidden_states = pca.transform(
                sampled_next_hidden_states.reshape(len(random_subset_indices), -1))

            displacement_vector = projected_sampled_next_hidden_states - projected_sampled_hidden_states

            vector_fields_by_side_by_stimuli[side][possible_stimulus] = dict(
                xrange=xrange,
                yrange=yrange,
                displacement_vector=displacement_vector,
                random_subset_indices=random_subset_indices,
                sampled_hidden_states=sampled_hidden_states,
                projected_sampled_hidden_states=projected_sampled_hidden_states,
                sampled_next_hidden_states=sampled_next_hidden_states,
                projected_sampled_next_hidden_states=projected_sampled_next_hidden_states)

    return vector_fields_by_side_by_stimuli


def compute_model_fixed_points(model,
                               trial_data,
                               hidden_states,
                               num_grad_steps=10):

    assert num_grad_steps > 0

    # project all hidden states using pca
    # reshape to (num trials, num layers * hidden dimension)
    projected_hidden_states, xrange, yrange, pca = compute_projected_hidden_states_pca(
        hidden_states=hidden_states.reshape(hidden_states.shape[0], -1))

    # identify non-first block indices
    possible_stimuli = np.linspace(-1.5, 1.5, 3)
    fixed_points_by_side_by_stimuli = {}
    for side, trial_data_preferred_side in trial_data.groupby('stimuli_preferred_sides'):
        fixed_points_by_side_by_stimuli[side] = dict()

        for possible_stimulus in possible_stimuli:
            non_first_block_indices = trial_data_preferred_side.index[
                trial_data_preferred_side['stimuli_block_number'] != 1].to_numpy()

            # sample subset of indices
            random_subset_indices = np.random.choice(
                non_first_block_indices,
                replace=False,
                size=len(non_first_block_indices))

            initial_sampled_hidden_states = torch.from_numpy(
                hidden_states[random_subset_indices])
            projected_initial_sampled_hidden_states = projected_hidden_states[random_subset_indices]
            fixed_point_hidden_states = initial_sampled_hidden_states.clone().requires_grad_(True)

            rewards = torch.from_numpy(
                trial_data.iloc[random_subset_indices]['rewards'].to_numpy()).reshape(-1, 1)

            stimuli = torch.zeros(
                size=(len(random_subset_indices), 1, 1)).fill_(possible_stimulus)

            optimizer = torch.optim.SGD([fixed_point_hidden_states], lr=0.01)
            print(f'Finding fixed points using {num_grad_steps} gradient steps')
            for _ in range(num_grad_steps):
                optimizer.zero_grad()
                model_forward_output = run_model_one_step(
                    model=model,
                    stimuli=stimuli,
                    rewards=rewards,
                    hidden_states=fixed_point_hidden_states)
                displacement_vector = model_forward_output['core_hidden'] - fixed_point_hidden_states
                displacement_vector_norm = torch.norm(displacement_vector, dim=(1, 2))
                # print(f'Norm of smallest displacement vector: {torch.min(displacement_vector)}')
                # normalized_velocity = displacement_vector / torch.norm(sampled_hidden_states, dim=2).squeeze(1)
                loss = torch.mean(displacement_vector_norm)
                loss.backward()
                optimizer.step()

            projected_final_sampled_hidden_states = pca.fit_transform(
                fixed_point_hidden_states.reshape(len(random_subset_indices), -1).detach().numpy())
            projected_displacement_vector = pca.fit_transform(
                displacement_vector.reshape(len(random_subset_indices), -1).detach().numpy())
            fixed_points_by_side_by_stimuli[side][possible_stimulus] = dict(
                xrange=xrange,
                yrange=yrange,
                projected_displacement_vector=projected_displacement_vector,
                random_subset_indices=random_subset_indices,
                initial_sampled_hidden_states=initial_sampled_hidden_states,
                projected_initial_sampled_hidden_states=projected_initial_sampled_hidden_states,
                final_sampled_hidden_states=fixed_point_hidden_states,
                projected_final_sampled_hidden_states=projected_final_sampled_hidden_states)

    return fixed_points_by_side_by_stimuli


def compute_projected_hidden_states_pca(hidden_states):
    # project all hidden states to 2 dimensions
    assert len(hidden_states.shape) == 2
    pca = PCA(n_components=2)
    pca.fit(hidden_states)
    projected_hidden_states = pca.fit_transform(hidden_states)
    min_x, max_x = min(projected_hidden_states[:, 0]), max(projected_hidden_states[:, 0])
    min_y, max_y = min(projected_hidden_states[:, 1]), max(projected_hidden_states[:, 1])
    return projected_hidden_states, (min_x, max_x), (min_y, max_y), pca


def compute_psytrack_fit(trial_data):

    psytrack_model_choice = trial_data['actions_chosen'].values
    if np.mean(psytrack_model_choice) > 0.95:
        print('Model made only one action')
        return
    psytrack_stimuli = trial_data['stimuli'].values.reshape(-1, 1)

    # psytrack inputs need to be shaped (N, M), where N is number of trials and
    # M is arbitrary integer
    psytrack_inputs = dict(
        s1=psytrack_stimuli)

    psytrack_data = dict(
        y=psytrack_model_choice,
        inputs=psytrack_inputs,
        name='Temp')
    weights_dict = dict(
        bias=1,
        s1=1)  # only fit first column (there is only 1 column)
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

    # except RuntimeError as e:
    #     # matrix is singular b/c model outputs all same actions
    #     # just continue
    #     return



def run_model_one_step(model,
                       stimuli,
                       rewards,
                       hidden_states=None):

    rewards = rewards.double()
    stimuli = stimuli.double()

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

    # creat single step model input
    model_input = dict(
        stimulus=stimuli,
        reward=rewards)

    model_forward_output = model(model_input)

    return model_forward_output
