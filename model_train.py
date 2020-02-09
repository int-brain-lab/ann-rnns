from datetime import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.env import create_biased_choice_worlds
from utils.hooks import create_hook_fns_train
from utils.run import create_model, create_optimizer, run_envs


def main():

    model = create_model(
        model_str='rnn',
        model_kwargs=dict(core_kwargs=dict(num_layers=1, hidden_size=10),
                          param_init='eye'))

    # model = create_model(
    #     model_str='ff',
    #     model_kwargs=dict(ff_kwargs=dict(activation_str='relu',
    #                                      layer_widths=[10, 10])))

    log_dir = os.path.join('runs', model.description_str + '_' + str(datetime.now()))
    tensorboard_writer = SummaryWriter(log_dir=log_dir)

    optimizer = create_optimizer(
        model=model,
        optimizer_str='sgd')

    envs = create_biased_choice_worlds(
        num_env=15,
        tensorboard_writer=tensorboard_writer)

    start_grad_step = 0
    num_grad_steps = 751

    hook_fns = create_hook_fns_train(
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    train_model_output = train_model(
        model=model,
        envs=envs,
        optimizer=optimizer,
        hook_fns=hook_fns,
        tensorboard_writer=tensorboard_writer,
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    tensorboard_writer.close()


def train_model(model,
                envs,
                optimizer,
                hook_fns,
                tensorboard_writer,
                start_grad_step=0,
                num_grad_steps=150,
                tag_prefix='train/'):

    # sets the model in training mode.
    model.train()

    # ensure assignment before reference
    run_envs_output = {}
    grad_step = start_grad_step

    for grad_step in range(start_grad_step, start_grad_step + num_grad_steps):
        optimizer.zero_grad()
        if hasattr(model, 'reset_core_hidden'):
            model.reset_core_hidden()
        avg_reward, avg_correct_choice, run_envs_output = run_envs(
            model=model,
            envs=envs)
        loss = -avg_reward
        loss.backward()
        optimizer.step()

        if grad_step in hook_fns:

            hook_input = dict(
                loss=loss.item(),
                avg_correct_choice=avg_correct_choice.item(),
                run_envs_output=run_envs_output,
                grad_step=grad_step,
                model=model,
                envs=envs,
                optimizer=optimizer,
                tensorboard_writer=tensorboard_writer,
                tag_prefix=tag_prefix)

            for hook_fn in hook_fns[grad_step]:
                hook_fn(hook_input)

    train_model_output = dict(
        grad_step=grad_step,
        run_envs_output=run_envs_output
    )

    return train_model_output


if __name__ == '__main__':
    torch.manual_seed(1)
    log_dir = 'runs'
    os.makedirs(log_dir, exist_ok=True)
    main()
