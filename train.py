import numpy as np

from utils.analysis import compute_eigenvalues
import utils.run


def train():

    setup_results = utils.run.setup_train()

    train_model(
        model=setup_results['model'],
        envs=setup_results['envs'],
        optimizer=setup_results['optimizer'],
        fn_hook_dict=setup_results['fn_hook_dict'],
        params=setup_results['params'],
        tensorboard_writer=setup_results['tensorboard_writer'])

    setup_results['tensorboard_writer'].close()


def train_model(model,
                envs,
                optimizer,
                fn_hook_dict,
                params,
                tensorboard_writer,
                tag_prefix='train/'):

    # sets the model in training mode.
    model.train()

    # ensure assignment before reference
    run_envs_output = {}
    start = params['run']['start_grad_step']
    stop = start + params['run']['num_grad_steps']

    for grad_step in range(start, stop):
        if hasattr(model, 'apply_connectivity_masks'):
            model.apply_connectivity_masks()
        if hasattr(model, 'reset_core_hidden'):
            model.reset_core_hidden()
        optimizer.zero_grad()
        run_envs_output = utils.run.run_envs(
            model=model,
            envs=envs)
        run_envs_output['avg_loss_per_dt'].backward()
        optimizer.step()

        if grad_step in fn_hook_dict:

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
                grad_step=grad_step,
                model=model,
                envs=envs,
                optimizer=optimizer,
                tensorboard_writer=tensorboard_writer,
                tag_prefix=tag_prefix,
                params=params)

            eigenvalues_svd_results = compute_eigenvalues(
                matrix=hook_input['hidden_states'].reshape(hook_input['hidden_states'].shape[0], -1))

            hook_input.update(eigenvalues_svd_results)

            for hook_fn in fn_hook_dict[grad_step]:
                hook_fn(hook_input)

    train_model_output = dict(
        grad_step=grad_step,
        run_envs_output=run_envs_output
    )

    return train_model_output


if __name__ == '__main__':
    train()
