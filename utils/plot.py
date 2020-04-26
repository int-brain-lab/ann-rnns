import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from psytrack.plot.analysisFunctions import makeWeightPlot
import scipy.cluster.hierarchy as spc
from scipy.stats import norm
import seaborn as sns

import utils.analysis

# increase resolution
plt.rcParams['figure.dpi'] = 200.
plt.rcParams['font.size'] = 4


def create_rotation_matrix(theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return rotation_matrix


rotation_matrix_90 = create_rotation_matrix(theta=np.pi / 2)

# map for converting left and right to numeric -1, 1 and vice versa
side_string_map = {
    'left': -1,
    -1: 'Left',
    'right': 1,
    1: 'Right'
}

side_color_map = {
    'left': 'tab:orange',
    side_string_map['left']: 'tab:orange',
    'right': 'tab:blue',
    side_string_map['right']: 'tab:blue',
    'neutral': 'tab:gray',
    'correct': 'tab:green',
    'incorrect': 'tab:red',
    'timeout': 'tab:purple',
    'ideal': 'k'
}


def hook_plot_analysis_psytrack_fit(hook_input):
    session_data = hook_input['session_data']

    trial_end_data = session_data[session_data.trial_end == 1]

    try:
        psytrack_fit_output = utils.analysis.compute_psytrack_fit(
            session_data=trial_end_data)
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
    #     END=len(trial_end_data),
    #     errorbar=credibleInt,
    #     perf_plot=True,
    #     bias_plot=True)

    # create subplots
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    num_trials_to_display = 500
    trial_num = np.arange(num_trials_to_display) + 1
    fig.suptitle(f'Bernoulli GLM Model (Psytrack by Roy & Pillow) (Num Points={len(trial_end_data)})')
    axes[3].set_xlabel('Trial Number')

    # plot stimuli values
    axes[0].plot(
        trial_num,
        trial_end_data['stimuli'].values[:num_trials_to_display],
        label='Stimulus Value')
    axes[1].set_ylabel('Stimulus Value')

    # plot block structure i.e. preferred side
    axes[1].plot(
        trial_num,
        trial_end_data['stimuli_preferred_sides'].values[:num_trials_to_display],
        label='Block Preferred Side')
    axes[1].scatter(
        trial_num,
        1.05 * trial_end_data['stimuli_sides'].values[:num_trials_to_display],
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
    axes[3].set_ylabel('BernGLM Feedback Weight')
    hook_input['tensorboard_writer'].add_figure(
        tag='psytrack_fit',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_dts_per_trial_by_stimuli_strength(hook_input):
    session_data = hook_input['session_data']
    dts_and_stimuli_strength_by_trial_df = session_data.groupby([
        'session_index', 'block_index', 'trial_index']).agg({
        'trial_strength': 'first',
        'rnn_step_index': 'size'})

    # plot trial number within block (x) vs probability of correct response (y)
    avg_dts_per_trial = dts_and_stimuli_strength_by_trial_df.groupby(
        ['trial_strength']).rnn_step_index.mean()
    sem_dts_per_trial = dts_and_stimuli_strength_by_trial_df.groupby(
        ['trial_strength']).rnn_step_index.sem()
    stimuli_strengths = avg_dts_per_trial.index.values

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Trial Strength')
    ax.set_ylabel('dts/trial')
    fig.suptitle('dts/Trial by Trial Stimulus Strength')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    ax.plot(
        stimuli_strengths,
        avg_dts_per_trial,
        color=side_color_map['neutral'])
    ax.fill_between(
        x=stimuli_strengths,
        y1=avg_dts_per_trial - sem_dts_per_trial,
        y2=avg_dts_per_trial + sem_dts_per_trial,
        alpha=0.3,
        linewidth=0,
        color=side_color_map['neutral'])

    hook_input['tensorboard_writer'].add_figure(
        tag='behav_dts_per_trial_by_stimuli_strength',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_trial_outcome_by_trial_strength(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    last_dt_within_trial_data = session_data[session_data.trial_end == 1.]

    trial_outcome_by_trial_strength = last_dt_within_trial_data.groupby(['signed_trial_strength']).agg({
        'block_index': 'size',  # can use any column to count number of datum in each group
        'action_taken': 'sum',
        'correct_action_taken': 'sum'
    })

    trial_outcome_by_trial_strength.rename(
        columns={'block_index': 'num_trials'},
        inplace=True)

    trial_outcome_by_trial_strength['timeout'] = \
        trial_outcome_by_trial_strength['num_trials'] - trial_outcome_by_trial_strength['action_taken']

    trial_outcome_by_trial_strength['incorrect_action_taken'] = \
        trial_outcome_by_trial_strength['action_taken'] - \
        trial_outcome_by_trial_strength['correct_action_taken']

    # drop unnecessary columns
    trial_outcome_by_trial_strength.drop(
        columns=['num_trials', 'action_taken'],
        inplace=True)

    # divide each row by row sum to get percents
    trial_outcome_by_trial_strength = trial_outcome_by_trial_strength.div(
        trial_outcome_by_trial_strength.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.suptitle('Trial Outcome (%) by Trial Stimulus Strength')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_ylabel('Trial Outcome (%)')
    ax.set_xlabel('Trial Stimulus Strength')
    ax.set_ylim([-0.05, 1.05])

    width = 0.35

    # add stacked percents
    ax.bar(trial_outcome_by_trial_strength.index,
           trial_outcome_by_trial_strength.timeout,
           width=width,
           label='Timeout',
           color=side_color_map['timeout'])

    ax.bar(trial_outcome_by_trial_strength.index,
           trial_outcome_by_trial_strength.incorrect_action_taken,
           width=width,
           bottom=trial_outcome_by_trial_strength.timeout,
           label='Incorrect Action',
           color=side_color_map['incorrect'])

    ax.bar(trial_outcome_by_trial_strength.index,
           trial_outcome_by_trial_strength.correct_action_taken,
           width=width,
           bottom=trial_outcome_by_trial_strength.timeout +
                  trial_outcome_by_trial_strength.incorrect_action_taken,
           label='Correct Action',
           color=side_color_map['correct'])

    ax.legend()

    hook_input['tensorboard_writer'].add_figure(
        tag='behav_trial_outcome_by_trial_strength',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_action_by_dts_within_trial(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    trial_end_data = session_data[session_data.trial_end == 1.].copy()

    # subtract the blank dts to correctly count number of observations
    trial_end_data.rnn_step_index -= hook_input['envs'][0].rnn_steps_before_stimulus - 1
    trial_end_data.loc[trial_end_data.rnn_step_index < 1, 'rnn_step_index'] = 0.

    avg_model_correct_action_prob_by_num_dts = trial_end_data.groupby(
        ['rnn_step_index'])['correct_action_taken'].mean()
    sem_model_correct_action_prob_by_num_dts = trial_end_data.groupby(
        ['rnn_step_index'])['correct_action_taken'].sem()

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.suptitle('Correct Action Trials / Total Trials by Number of Stimuli Within Trial')
    ax.set_xlabel('Number of Stimuli Within Trial')
    ax.set_ylabel('Correct Action Trials / Total Trials')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_ylim([-0.5, 1.05])
    ax.set_xlim([0., 1 + hook_input['envs'][0].max_stimuli_per_trial])

    dts_per_trial = avg_model_correct_action_prob_by_num_dts.index.values
    assert np.all(dts_per_trial > -1)
    ax.plot(
        dts_per_trial,
        avg_model_correct_action_prob_by_num_dts,
        '-o',
        label='Model',
        color=side_color_map['correct'])

    ax.fill_between(
        x=dts_per_trial,
        y1=avg_model_correct_action_prob_by_num_dts - sem_model_correct_action_prob_by_num_dts,
        y2=avg_model_correct_action_prob_by_num_dts + sem_model_correct_action_prob_by_num_dts,
        alpha=0.3,
        linewidth=0,
        color=side_color_map['correct'])

    ideal_prob_correct_after_dt_per_trial_weighted = pd.DataFrame(
        np.nan,  # initialize all to nan
        index=dts_per_trial,
        columns=trial_end_data.trial_strength.unique(),
        dtype=np.float16)
    for mu, number_of_trials in trial_end_data.groupby(['trial_strength']).size().iteritems():
        ideal_prob_correct_after_dt_per_trial_given_mu = np.array([
            1 - norm.cdf(-mu * np.sqrt(dt_per_trial)) if dt_per_trial > 0 else 0.5
            for dt_per_trial in dts_per_trial])
        fraction_of_trials = number_of_trials / len(trial_end_data)
        ideal_prob_correct_after_dt_per_trial_weighted[mu] = fraction_of_trials * \
                                                             ideal_prob_correct_after_dt_per_trial_given_mu

    # average over possible trial strengths
    ideal_prob_correct_after_dt_per_trial = ideal_prob_correct_after_dt_per_trial_weighted.sum(axis=1)

    ax.plot(
        dts_per_trial,
        ideal_prob_correct_after_dt_per_trial,
        '-o',
        label='Ideal P(Correct|Number of Stimuli)',
        color=side_color_map['ideal'])

    time_delay_penalty = hook_input['envs'][0].time_delay_penalty
    ideal_reward_rate_after_dt_per_trial = 2 * ideal_prob_correct_after_dt_per_trial - 1
    ideal_reward_rate_after_dt_per_trial -= (ideal_prob_correct_after_dt_per_trial.index *
                                             time_delay_penalty)

    ax.plot(
        dts_per_trial,
        ideal_reward_rate_after_dt_per_trial,
        '-d',
        label=f'Ideal Reward Rate (Delay Penalty: {time_delay_penalty})',
        color=side_color_map['ideal'])

    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_action_by_dts_per_trial',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_action_by_trial_within_block(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    last_dt_within_trial_data = session_data[session_data.trial_end == 1.]

    # plot trial number within block (x) vs probability of correct response (y)
    # TODO: use the feedback, not the model's probability action
    avg_model_correct_action_prob_by_trial_num = last_dt_within_trial_data.groupby(
        ['trial_index'])['correct_action_taken'].mean()
    sem_model_correct_action_prob_by_trial_num = last_dt_within_trial_data.groupby(
        ['trial_index'])['correct_action_taken'].sem()

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.suptitle('Correct Action Trials / Total Trials by Trial Within Block')
    ax.set_xlabel('Trial Within Block')
    ax.set_ylabel('Correct Action Trials / Total Trials')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0., 101.])
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    ax.plot(
        1 + avg_model_correct_action_prob_by_trial_num.index.values,
        avg_model_correct_action_prob_by_trial_num,
        '-o',
        color=side_color_map['neutral'])

    ax.fill_between(
        x=1 + avg_model_correct_action_prob_by_trial_num.index.values,
        y1=avg_model_correct_action_prob_by_trial_num - sem_model_correct_action_prob_by_trial_num,
        y2=avg_model_correct_action_prob_by_trial_num + sem_model_correct_action_prob_by_trial_num,
        alpha=0.3,
        linewidth=0,
        color=side_color_map['neutral'])

    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_action_by_trial_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_action_on_block_side_trial_side_by_trial_strength(hook_input):
    session_data = hook_input['session_data']

    # keep only last dts in trials
    last_dt_within_trial_data = session_data[session_data.trial_end == 1.]

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.suptitle('Correct Action Trials / Total Trials by Trial Stimulus Strength')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_ylabel('Correct Action Trials / Total Trials')
    ax.set_xlabel('Trial Stimulus Strength')

    for (block_side, trial_side), block_side_trial_side_data in last_dt_within_trial_data.groupby([
        'block_side', 'trial_side']):
        avg_correct_action_prob_by_stim_strength = block_side_trial_side_data.groupby(
            'trial_strength')['correct_action_taken'].mean()

        sem_correct_action_prob_by_stim_strength = block_side_trial_side_data.groupby(
            'trial_strength')['correct_action_taken'].sem()

        block_side_str = side_string_map[block_side]
        trial_side_str = side_string_map[trial_side]

        ax.plot(
            avg_correct_action_prob_by_stim_strength.index,
            avg_correct_action_prob_by_stim_strength,
            '-o' if block_side == trial_side else '--o',
            # solid lines for consistent block side, trial side; dotted otherwise
            label=f'{block_side_str} Block, {trial_side_str} Trial',
            color=side_color_map[block_side])
        ax.fill_between(
            x=avg_correct_action_prob_by_stim_strength.index,
            y1=avg_correct_action_prob_by_stim_strength - sem_correct_action_prob_by_stim_strength,
            y2=avg_correct_action_prob_by_stim_strength + sem_correct_action_prob_by_stim_strength,
            alpha=0.3,
            linewidth=0,
            color=side_color_map[block_side])

    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_action_on_block_side_trial_side_by_trial_strength',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_prob_correct_slope_intercept_by_prev_block_duration(hook_input):
    session_data = hook_input['session_data']

    # only take consider last dt within a trial
    session_data = session_data[session_data['trial_end'] == 1]

    num_trials_to_consider = 4

    new_data = dict(
        prev_block_durations=[],
        model_prob_correct_slopes=[],
        model_prob_correct_intercepts=[])

    # TODO: can this be refactored using multiple aggregate?
    for (session_index, block_index), block_session_data in session_data.groupby(
            ['session_index', 'block_index']):

        # skip first block because they have no preceding block!
        if block_index == 0:
            continue

        # skip truncated blocks with fewer than the minimum number of trials
        if len(block_session_data) < num_trials_to_consider:
            continue

        # keep only the first ten trials
        first_n_trials = block_session_data[block_session_data.trial_index < num_trials_to_consider]

        prev_block_duration = max(
            session_data[(session_data.session_index == session_index) &
                         (session_data.block_index == (block_index - 1))].trial_index)

        # calculate slope of best fit
        # TODO: this is the wrong regression
        coefficients = np.polyfit(
            x=first_n_trials.trial_index.values.astype(np.float32),
            y=first_n_trials.correct_action_taken.values.astype(np.float32),  # need to convert
            deg=1)
        slope, intercept = coefficients[0], coefficients[1]

        # plot the best fit line
        # plt.plot(first_n_trials.trial_index.values.astype(np.float32),
        #          first_n_trials.correct_action_prob.values.astype(np.float32))
        # plt.plot(first_n_trials.trial_index.values.astype(np.float32),
        #          np.poly1d(coefficients)(first_n_trials.trial_index.values.astype(np.float32)))

        new_data['prev_block_durations'].append(prev_block_duration)
        new_data['model_prob_correct_slopes'].append(slope)
        new_data['model_prob_correct_intercepts'].append(intercept)

    means_sem = pd.DataFrame(new_data).groupby('prev_block_durations').agg(['mean', 'sem'])

    fig, axes = plt.subplots(nrows=2, figsize=(4, 3), sharex=True)
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for i, column_str in enumerate(['model_prob_correct_slopes', 'model_prob_correct_intercepts']):
        ax = axes[i]
        if i == 0:
            ax.set_title(f'Slope of P(Correct Action) (First {num_trials_to_consider}'
                         f' Trials) by Previous Block Duration')
            ax.set_ylabel('Slope of Model P(Correct Action)')
        elif i == 1:
            ax.set_title(f'Intercept of P(Correct Action) (First {num_trials_to_consider}'
                         f' Trials) by Previous Block Duration')
            ax.set_ylabel('Intercept of Model P(Correct Action)')
        else:
            raise ValueError('Impermissible axis number')

        # plot mean
        ax.plot(means_sem.index.values,
                means_sem[column_str]['mean'].values,
                '-o',
                markersize=2,
                color=side_color_map['neutral'])

        # add SEM
        ax.fill_between(
            x=means_sem.index.values,
            y1=means_sem[column_str]['mean'].values - means_sem[column_str]['sem'].values,
            y2=means_sem[column_str]['mean'].values + means_sem[column_str]['sem'].values,
            alpha=0.3,
            linewidth=0,
            color=side_color_map['neutral'])
    ax.set_xlabel('Previous Block Duration')
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_prob_correct_slope_intercept_by_prev_block_duration',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_subj_prob_block_switch_by_signed_trial_strength(hook_input):
    session_data = hook_input['session_data']

    trial_data = pd.DataFrame(
        columns=['signed_trial_strength',
                 'left_action_prob',
                 'reward',
                 '1_back_correct',
                 '2_back_correct',
                 '3_back_correct',
                 'prev_left_action_prob',
                 'subjective_block_switch'],
        dtype=np.float16)

    for session_index, per_session_data in session_data.groupby(['session_index']):
        # TODO lambda breaks if len(x) < 2
        trial_data_within_session = per_session_data.groupby(['block_index', 'trial_index']).agg({
            'signed_trial_strength': 'first',  # arbitrary
            'left_action_prob': 'first',
            'reward': 'last',
            '1_back_correct': 'last',
            '2_back_correct': 'last',
            '3_back_correct': 'last'})

        trial_data_within_session['prev_left_action_prob'] = trial_data_within_session.left_action_prob.shift(periods=1)

        # drop rows with no previous action
        trial_data_within_session = trial_data_within_session[
            ~pd.isna(trial_data_within_session.prev_left_action_prob)]

        trial_data_within_session['subjective_block_switch'] = (trial_data_within_session.prev_left_action_prob < 0.5) ^ \
                                                               (trial_data_within_session.left_action_prob < 0.5)

        trial_data = trial_data.append(trial_data_within_session)

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.suptitle('Subjective P(Block Switch) by Signed Trial Strength')
    ax.set_ylim([-0.05, 1.05])
    max_trial_strength = max(hook_input['envs'][0].possible_trial_strengths)
    ax.set_xlim([-max_trial_strength, max_trial_strength])
    ax.set_xlabel('Signed Trial Strength')
    ax.set_ylabel('Subjective P(Block Switch)')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    try:
        mean_prob_block_switch_reward = trial_data[trial_data.reward == 1].groupby(['signed_trial_strength'])[
            'subjective_block_switch'].mean()
        ax.plot(
            mean_prob_block_switch_reward.index.values,
            mean_prob_block_switch_reward,
            '-o',
            linewidth=1,
            markersize=5,
            fillstyle='none',
            label='Reward',
            color=side_color_map['neutral'])
    except pd.core.base.DataError:
        pass

    try:
        avg_prob_block_switch_1_back = trial_data[
            (trial_data.reward == -1) & (trial_data['1_back_correct'] == 1)].groupby([
            'signed_trial_strength'])['subjective_block_switch'].mean()
        ax.plot(
            avg_prob_block_switch_1_back.index.values,
            avg_prob_block_switch_1_back,
            '-+',
            linewidth=1,
            markersize=5,
            label='1-Back Error',
            color=side_color_map['neutral'])
    except pd.core.base.DataError:
        pass

    try:
        avg_prob_block_switch_2_back = trial_data[
            (trial_data.reward == -1) & (trial_data['1_back_correct'] == -1) & (
                    trial_data['2_back_correct'] == 1)].groupby([
            'signed_trial_strength'])['subjective_block_switch'].mean()
        ax.plot(
            avg_prob_block_switch_2_back.index.values,
            avg_prob_block_switch_2_back,
            '-d',
            linewidth=1,
            markersize=5,
            label='2-Back Error',
            color=side_color_map['neutral'])
    except pd.core.base.DataError:
        pass

    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_subj_prob_block_switch_by_signed_trial_strength',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_behav_right_action_after_error_by_right_action_after_correct(hook_input):
    # TODO: Behavioral paper 4g
    print(10)
    pass


def hook_plot_behav_right_action_by_signed_contrast(hook_input):
    session_data = hook_input['session_data']

    # only take consider last dt within a trial
    action_data = session_data[session_data['action_taken'] == 1]

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.suptitle('Right Action Taken / Total Action Trials by Signed Trial Strength')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Signed Trial Strength')
    ax.set_ylabel('Right Action Taken / Total Action Trials')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    mean_sem_by_signed_trial_strength = action_data.groupby(
        ['block_side', 'signed_trial_strength']).agg(
        {'action_side': ['mean', 'sem']})['action_side']

    # rescale from [-1, -1] to [0, 1]
    mean_sem_by_signed_trial_strength['mean'] = (1 + mean_sem_by_signed_trial_strength['mean']) / 2
    mean_sem_by_signed_trial_strength['sem'] = mean_sem_by_signed_trial_strength['sem'] / 2

    for block_side in action_data['block_side'].unique():
        # take cross section of block side
        mean_sem_by_signed_trial_strength_by_block_side = \
            mean_sem_by_signed_trial_strength.xs(block_side)

        # plot non-block conditioned
        ax.plot(
            mean_sem_by_signed_trial_strength_by_block_side.index.values,
            mean_sem_by_signed_trial_strength_by_block_side['mean'],
            '-o',
            label=side_string_map[block_side] + ' Block',
            color=side_color_map[block_side])

        ax.fill_between(
            x=mean_sem_by_signed_trial_strength_by_block_side.index.values,
            y1=mean_sem_by_signed_trial_strength_by_block_side['mean'] -
               mean_sem_by_signed_trial_strength_by_block_side['sem'],
            y2=mean_sem_by_signed_trial_strength_by_block_side['mean'] +
               mean_sem_by_signed_trial_strength_by_block_side['sem'],
            alpha=0.3,
            linewidth=0,
            color=side_color_map[block_side])

    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='behav_rightward_action_by_signed_contrast',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_hidden_to_hidden_jacobian_eigenvalues_complex_plane(hook_input):
    fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']

    # plot each fixed point in phase space

    jacobians_by_side_by_stimuli = utils.analysis.compute_jacobians_by_side_by_stimuli(
        model=hook_input['model'],
        trial_data=hook_input['session_data'],
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
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_hidden_to_hidden_jacobian_time_constants(hook_input):
    fixed_points_by_side_by_stimuli = hook_input['fixed_points_by_side_by_stimuli']

    # plot each fixed point in phase space

    jacobians_by_side_by_stimuli = utils.analysis.compute_jacobians_by_side_by_stimuli(
        model=hook_input['model'],
        trial_data=hook_input['session_data'],
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
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_hidden_unit_correlations(hook_input):
    # hidden states shape: (num rnn steps, num layers, hidden dimension)
    hidden_states = hook_input['hidden_states']
    hidden_size = hidden_states.shape[2]

    # reshape to (num trials, num layers * hidden dimension)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    trial_side = np.expand_dims(hook_input['session_data'].trial_side.values, 1)
    trial_side_orthogonal = np.expand_dims(hook_input['session_data'].trial_side_orthogonal.values, 1)
    block_side = np.expand_dims(hook_input['session_data'].block_side.values, 1)
    feedback = np.expand_dims(hook_input['session_data'].reward.values, 1)

    # construct correlation matrix
    hidden_states_and_task_variables = np.hstack((
        hidden_states,
        trial_side,
        trial_side_orthogonal,
        block_side,
        feedback))
    hidden_states_and_task_variables_correlations = np.corrcoef(hidden_states_and_task_variables.T)
    # due to machine error, correlation matrix isn't exactly symmetric (typically has e-16 errors)
    # make it symmetric
    hidden_states_and_task_variables_correlations = (hidden_states_and_task_variables_correlations +
                                                     hidden_states_and_task_variables_correlations.T) / 2
    hidden_state_self_correlations = hidden_states_and_task_variables_correlations[:hidden_size, :hidden_size]
    hidden_state_task_correlations = hidden_states_and_task_variables_correlations[:hidden_size, hidden_size:]

    # compute pairwise distances
    pdist = spc.distance.pdist(hidden_state_self_correlations)
    linkage = spc.linkage(pdist, method='complete')
    labels = spc.fcluster(linkage, 0.5 * np.max(pdist), 'distance')
    indices = np.argsort(labels)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(9, 3),
        gridspec_kw={"width_ratios": [1, 0.45, 1, 0.45]})
    # recurrent_mask_str = hook_input['model'].model_kwargs['connectivity_kwargs']['recurrent_mask']
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    # plot hidden state correlations
    ax = axes[0]
    sns.heatmap(hidden_state_self_correlations[indices][:, indices],
                cmap='RdBu_r',
                ax=ax,
                center=0,
                vmin=-1.,
                vmax=1.,
                square=True,
                xticklabels=indices,  # indices
                yticklabels=indices,  # indices
                cbar_kws={'label': 'Correlation', 'shrink': 0.5})
    ax.set_title('Hidden Unit - Hidden Unit Correlations')
    ax.set_xlabel('Hidden Unit Number')
    ax.set_ylabel('Hidden Unit Number')
    ax.set_aspect("equal")  # ensures little squares don't become rectangles

    if hook_input['tag_prefix'] == 'analyze/':
        ax = axes[1]
        normalized_readout_vectors = np.concatenate(
            [hook_input['trial_readout_vector'],
             hook_input['block_readout_vector']],
            axis=0)
        sns.heatmap(normalized_readout_vectors[:, indices].T,
                    cmap='RdBu_r',
                    ax=ax,
                    center=0,
                    xticklabels=['Trial Readout', 'Block Readout'],
                    yticklabels=indices,
                    square=True,
                    cbar_kws={'label': 'Correlation', 'shrink': 0.5})
        ax.set_title('Readout Vectors')
        ax.set_ylabel('Hidden Unit Number')

    recurrent_matrix = hook_input['model'].core.weight_hh_l0.data.numpy()
    dimension_ratio = recurrent_matrix.shape[0] / recurrent_matrix.shape[1]
    # # RNN weight will have shape (hidden size, hidden size)
    if dimension_ratio == 1:
        recurrent_matrix = recurrent_matrix[indices][:, indices]
    # LSTM weight will have shape (4*hidden size, hidden_size)
    # GRU weight will have shape (3*hidden size, hidden size)
    elif dimension_ratio == 4 or dimension_ratio == 3:
        pass
        # TODO add recurrent weight
    #     # TODO unknown whether this is correct
    #     for i in range(int(dimension_ratio)):
    #         recurrent_matrix[i*hidden_size:(i+1)*hidden_size] = \
    #             recurrent_matrix[i*hidden_size + indices][:, indices]
    else:
        raise ValueError('Unknown dimension ratio for recurrent weight matrix')

    # plot recurrent matrix values
    ax = axes[2]
    sns.heatmap(recurrent_matrix,
                cmap='RdBu_r',
                ax=ax,
                center=0,
                xticklabels=indices,
                yticklabels=indices,
                square=True,
                cbar_kws={'label': 'Weight Strength', 'shrink': 0.5})
    ax.set_title('Recurrent Weight Strength')
    ax.set_xlabel('Hidden Unit Number')
    ax.set_ylabel('Hidden Unit Number')
    ax.set_aspect("equal")  # ensures little squares don't become rectangles

    # hidden state vs task side, block side correlation
    ax = axes[3]
    sns.heatmap(hidden_state_task_correlations[indices, :],
                cmap='RdBu_r',
                ax=ax,
                center=0,
                xticklabels=['Trial', 'Ortho Trial', 'Block', 'Feedback'],
                yticklabels=indices,
                square=True,
                cbar_kws={'label': 'Correlation', 'shrink': 0.5})
    ax.set_title('Hidden Unit - Task Correlations')
    ax.set_ylabel('Hidden Unit Number')

    hook_input['tensorboard_writer'].add_figure(
        tag='model_hidden_unit_correlations',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_hidden_unit_fraction_var_explained(hook_input):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.arange(1, 1 + len(hook_input['frac_variance_explained'])),
            hook_input['frac_variance_explained'],
            'o',
            alpha=0.8,
            ms=3,
            color=side_color_map['neutral'])
    fig.suptitle('Fraction of Cumulative Variance Explained by Dimension')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Fraction of Cumulative Variance Explained')
    ax.set_ylim([-0.05, 1.05])
    hook_input['tensorboard_writer'].add_figure(
        tag='var_explained_by_dimension',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_weights_and_gradients(hook_input):
    weights = dict(
        input=hook_input['model'].core.weight_ih_l0.data.numpy(),
        recurrent=hook_input['model'].core.weight_hh_l0.data.numpy(),
        readout=hook_input['model'].readout.weight.data.numpy().T  # transpose for better plotting
    )

    if hook_input['tag_prefix'] != 'analyze/':
        weight_gradients = dict(
            input=hook_input['model'].core.weight_ih_l0.grad.numpy(),
            recurrent=hook_input['model'].core.weight_hh_l0.grad.numpy(),
            readout=hook_input['model'].readout.weight.grad.numpy().T  # transpose for better plotting
        )

    fig, axes = plt.subplots(nrows=2,
                             ncols=4,  # rows, cols
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
                             figsize=(9, 6))
    recurrent_mask_str = hook_input['model'].model_kwargs['connectivity_kwargs']['recurrent_mask']
    fig.suptitle(f'Model Weights (Recurrent Mask: {recurrent_mask_str}) and Gradients')
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for i, weight_str in enumerate(weights):
        axes[0, i].set_title(f'{weight_str} Matrix')
        hm = sns.heatmap(
            weights[weight_str],
            cmap='RdBu_r',
            square=True,
            ax=axes[0, i],
            center=0,
            vmin=-0.5,
            vmax=0.5,
            cbar_ax=axes[0, -1],
            cbar_kws={'label': 'Weight Strength'})
        axes[0, i].set_aspect("equal")  # ensures little squares don't become rectangles

        if hook_input['tag_prefix'] != 'analyze/':
            axes[1, i].set_title(f'{weight_str} Gradient')
            hm = sns.heatmap(
                weight_gradients[weight_str],
                cmap='RdBu_r',
                square=True,
                ax=axes[1, i],
                center=0,
                vmin=-0.5,
                vmax=0.5,
                cbar_ax=axes[1, -1],
                cbar_kws={'label': 'Weight Strength'})
            axes[1, i].set_aspect("equal")  # ensures little squares don't become rectangles

    hook_input['tensorboard_writer'].add_figure(
        tag='model_weights_and_gradients',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_model_weights_community_detection(hook_input):
    utils.analysis.compute_model_weights_community_detection(hook_input['model'])

    print(10)


def hook_plot_state_space_fixed_points(hook_input):
    displacement_norm_cutoff = 0.5

    # TODO: deduplicate with vector fields plot
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
                ax.set_ylabel(f'{stimulus}')
            # else:
            #     ax.set_yticklabels([])

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

            add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    fig.suptitle(f'Fixed Points (Num Grad Steps = {num_grad_steps})')

    # merge the rightmost column for the colorbar
    gs = axes[0, 2].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(sc, cax=ax_colorbar)
    color_bar.set_label(r'$||h_t - RNN(h_t, s_t) ||_2$')
    hook_input['tensorboard_writer'].add_figure(
        tag='hook_plot_psytrack_fit',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_vector_fields(hook_input):
    # TODO: deduplicate with hook_plot_hidden_state_projected_fixed_points
    session_data = hook_input['session_data']

    vector_fields_by_side_by_stimuli = utils.analysis.compute_model_hidden_state_vector_field(
        model=hook_input['model'],
        session_data=session_data,
        hidden_states=hook_input['hidden_states'],
        pca=hook_input['pca'],
        pca_hidden_states=hook_input['pca_hidden_states'])

    num_stimuli = len(vector_fields_by_side_by_stimuli[1.0].keys())

    # for feedback, session_data_by_feedback in session_data.groupby(['reward']):

    fig, axes = plt.subplots(nrows=num_stimuli,
                             ncols=3,
                             gridspec_kw={"width_ratios": [1, 1, 0.05]},
                             figsize=(9, 6),
                             sharex=True,
                             sharey=True)
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)

    for c, (side, vector_fields_by_stimuli_dict) in \
            enumerate(vector_fields_by_side_by_stimuli.items()):

        for r, (stimulus, vector_field_dict) in enumerate(vector_fields_by_stimuli_dict.items()):

            ax = axes[r, c]
            ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
            ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
            if r == 0:
                ax.set_title(f'Block Side: {side_string_map[side]}')
            elif r == num_stimuli - 1:
                ax.set_xlabel('Principal Component #1')

            if c == 0:
                ax.set_ylabel(stimulus)
            # else:
            # ax.set_yticklabels([])

            vector_magnitude = np.linalg.norm(
                vector_field_dict['displacement_vector'],
                axis=1)

            qvr = ax.quiver(
                vector_field_dict['sampled_pca_hidden_states'][:, 0],
                vector_field_dict['sampled_pca_hidden_states'][:, 1],
                0.005 * vector_field_dict['displacement_vector'][:, 0] / vector_magnitude,
                0.005 * vector_field_dict['displacement_vector'][:, 1] / vector_magnitude,
                vector_magnitude,
                scale=.1,
                cmap='gist_rainbow')

            add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    # merge the rightmost column for the colorbar
    gs = axes[0, 2].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(qvr, cax=ax_colorbar)
    color_bar.set_label(r'$||h_t - RNN(h_t, s_t) ||_2$')
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_vector_fields',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_trajectories_between_trials(hook_input):
    session_data = hook_input['session_data']

    # keep only data with feedback
    session_data = session_data[(session_data.reward == -1.) | (session_data.reward == 1)]

    # drop the last dt because no subsequent dt. prevents IndexError
    session_data = session_data[:len(session_data) - 1]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(4, 3),
        gridspec_kw={"width_ratios": [1, 1, 0.05]})

    for i, ((feedback, trial_side), session_data_by_trial_side_and_feedback) in \
            enumerate(session_data.groupby(['reward', 'trial_side'])):
        
        row, col = int(i / 2), int(i % 2)
        ax = axes[row, col]
        ax.set_title(f'{side_string_map[trial_side]} Trials, Feedback: {feedback}')
        if row == 1:
            ax.set_xlabel('Principal Component #1')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('Principal Component #2')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])

        pre_feedback_indices = session_data_by_trial_side_and_feedback.index

        post_feedback_indices = pre_feedback_indices + 1

        pca_hidden_states_pre_feedback = hook_input['pca_hidden_states'][pre_feedback_indices]
        pca_hidden_states_post_feedback = hook_input['pca_hidden_states'][post_feedback_indices]

        displacement_vectors = pca_hidden_states_post_feedback - pca_hidden_states_pre_feedback
        displacement_norm = np.linalg.norm(displacement_vectors, axis=1)

        qvr = ax.quiver(
            pca_hidden_states_pre_feedback[:, 0],
            pca_hidden_states_pre_feedback[:, 1],
            0.005 * displacement_vectors[:, 0] / displacement_norm,
            0.005 * displacement_vectors[:, 1] / displacement_norm,
            displacement_norm,
            scale=.1,
            # alpha=0.4,
            cmap='gist_rainbow')

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    gs = axes[0, 2].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    ax_colorbar = fig.add_subplot(gs[:, -1])
    color_bar = fig.colorbar(qvr, cax=ax_colorbar)
    color_bar.set_label(r'$||h_t - RNN(h_t, f_t) ||_2$')

    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trajectories_between_trials',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_trajectories_within_block(hook_input):
    session_data = hook_input['session_data']

    # take only last dt within a trial
    # exclude blocks that are first in the session
    session_data = session_data[session_data.trial_end == 1]

    num_rows, num_cols = 2, 2

    # select only environment 0, last num_rows * num_cols blocks
    subset_session_data = session_data[(session_data.session_index == 0) &
                                       (session_data.block_index > max(session_data.block_index) - num_cols*num_rows)]
    # separate by side bias
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(7, 7))
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    # plt.suptitle('State Space Trajectories ({} degrees between readout vectors)'.format(
    #     hook_input['degrees_btwn_pca_trial_block_vectors']))

    # create possible color range
    max_block_duration = max(subset_session_data.groupby(['session_index', 'block_index']).size())

    for i, (block_idx, session_data_by_block) in enumerate(subset_session_data.groupby('block_index')):

        row, col = int(i / num_cols), int(i % num_cols)
        ax = axes[row, col]
        block_side = side_string_map[session_data_by_block.block_side.unique()[0]]
        ax.set_title(f'Block {1 + int(block_idx)}\n{block_side} Block')
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
        if row == (num_rows - 1):
            ax.set_xlabel('Principal Component #1')
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('Principal Component #2')
        else:
            ax.set_yticklabels([])

        block_indices = session_data_by_block.index.values
        proj_hidden_states_block = hook_input['pca_hidden_states'][block_indices]
        # stimuli = np.round(trial_data_by_block['stimuli'].values, 1)
        # segment_text = np.where(session_data_by_block['reward'] > 0.9, 'C', 'I')
        for i in range(len(block_indices) - 1):
            ax.plot(
                proj_hidden_states_block[i:i + 2, 0],
                proj_hidden_states_block[i:i + 2, 1],
                '-o',
                color=plt.cm.jet(i / max_block_duration),
                markersize=2,
                # linestyle='None',
                zorder=2)
            # ax.text(
            #     proj_hidden_states_block[i, 0],
            #     proj_hidden_states_block[i, 1],
            #     # str(stimuli[i]),
            #     segment_text[i]
            # )

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    # TODO: add colobar without disrupting
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max_block_duration))
    # color_bar = fig.colorbar(sm, cax=axes[-1])
    # color_bar.set_label('Trial Number within Block')
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trajectories_within_block',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_trajectories_within_block_smooth(hook_input):
    trajectory_controlled_output = utils.analysis.compute_projected_hidden_state_trajectory_controlled(
        model=hook_input['model'],
        pca=hook_input['pca'])

    session_data = trajectory_controlled_output['session_data']
    max_block_len = max(session_data.groupby(['session_index', 'stimuli_block_number']).size())

    fig, axes = plt.subplots(nrows=3,
                             ncols=4,  # 1 row, 3 columns
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]},
                             figsize=(18, 12))
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    plt.suptitle(f'Model State Space (Projected) Smooth Trajectories')

    for block_num, trial_data_by_block in session_data.groupby('stimuli_block_number'):
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
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_trajectories_within_trial(hook_input):
    session_data = hook_input['session_data']

    num_rows, num_cols = 3, 3
    # separate by side bias
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             gridspec_kw={"width_ratios": [1] * num_cols},
                             figsize=(4, 4))
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    # plt.suptitle('State Space PCA Trajectories ({} degrees between readout vectors)'.format(
    #     hook_input['degrees_btwn_pca_trial_block_vectors']))

    # select only environment 1, first 12 trials
    subset_session_data = session_data[(session_data['session_index'] == 0) &
                                       (session_data['block_index'] == 2) &
                                       (session_data['trial_index'] < num_cols * num_rows)]

    # create possible color range
    max_trial_duration = max(subset_session_data.groupby(['session_index', 'block_index', 'trial_index']).size())

    for trial_index, session_data_by_trial in subset_session_data.groupby('trial_index'):

        if trial_index >= num_cols * num_rows:
            break

        row, col = int(trial_index / num_cols), int(trial_index % num_cols)
        ax = axes[row, col]
        trial_side = side_string_map[session_data_by_trial.trial_side.unique()[0]]
        title = f'{trial_side} Trial, '
        title += 'Correct Action' if bool(session_data_by_trial.tail(1).iloc[0].correct_action_taken) \
            else 'Incorrect Action'
        ax.set_title(title)
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])


        if row == (num_rows - 1):
            ax.set_xlabel('Principal Component #1')
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel('Principal Component #2')
        else:
            ax.set_yticklabels([])

        trial_indices = session_data_by_trial.index.values
        proj_hidden_states_block = hook_input['pca_hidden_states'][trial_indices]

        # plot the first dt in the trial
        ax.plot(
            proj_hidden_states_block[0, 0],
            proj_hidden_states_block[0, 1],
            'o-',
            markersize=1,
            color=plt.cm.jet(0 / max_trial_duration),
            zorder=2)

        # plot the rest of the trial's dts
        for i in range(1, len(trial_indices)):
            ax.plot(
                proj_hidden_states_block[i - 1:i + 1, 0],
                proj_hidden_states_block[i - 1:i + 1, 1],
                'o-',
                markersize=1,
                color=plt.cm.jet(i / max_trial_duration))
            # ax.text(
            #     proj_hidden_states_block[i, 0],
            #     proj_hidden_states_block[i, 1],
            #     # str(stimuli[i]),
            #     segment_text[i]
            # )

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    # TODO: add colobar without disrupting
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max_block_duration))
    # color_bar = fig.colorbar(sm, cax=axes[-1])
    # color_bar.set_label('Trial Number within Block')
    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trajectories_within_trial',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_state_space_trials_by_classifier_and_trial_index(hook_input):
    session_data = hook_input['session_data']

    # take only last dt within a trial
    # exclude blocks that are first in the session
    trial_end_data = session_data[session_data.trial_end == 1]

    fig, axes = plt.subplots(nrows=1,
                             ncols=1,
                             gridspec_kw={"width_ratios": [1]},
                             figsize=(5, 5))

    max_trials_per_block_to_consider = 50
    trial_end_data = trial_end_data[trial_end_data.trial_index < max_trials_per_block_to_consider]

    titles = [
        # 'Colored by Trial Index',
        'Colored by Classifier'
    ]

    color_arrays = [
        # plt.cm.jet(trial_end_data.trial_index / max_trials_per_block_to_consider),
        plt.cm.jet((1. + trial_end_data.classifier_block_side) / 2)
    ]

    for row, (color_array, title) in enumerate(zip(color_arrays, titles)):

        ax = axes
        # ax = axes[row]
        ax.set_title(f'{title}')
        ax.set_xlim(hook_input['pca_xrange'][0], hook_input['pca_xrange'][1])
        ax.set_ylim(hook_input['pca_yrange'][0], hook_input['pca_yrange'][1])
        ax.set_xlabel('Principal Component #1')
        ax.set_ylabel('Principal Component #2')
        if row == 1:
            ax.set_xlabel('Principal Component #1')
        else:
            ax.xaxis.set_ticklabels([])

        block_side_trial_end_rows = trial_end_data.index.values
        block_side_trial_end_proj_hidden_states = hook_input['pca_hidden_states'][block_side_trial_end_rows]
        ax.scatter(
            block_side_trial_end_proj_hidden_states[:, 0],
            block_side_trial_end_proj_hidden_states[:, 1],
            alpha=0.4,
            s=1,
            c=color_array)

        add_pca_readout_vectors_to_axis(ax=ax, hook_input=hook_input)

    hook_input['tensorboard_writer'].add_figure(
        tag='state_space_trials_by_block_side',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_block_side_trial_side_by_trial_number(hook_input):
    session_data = hook_input['session_data']

    # keep only session 0
    session_data = session_data[session_data.session_index == 0]
    first_dt_of_each_trial = session_data.groupby(['block_index', 'trial_index']).first()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    ax.set_title('Block Side, Trial Side by Trial Number')
    ax.set_xlabel('Trial Number within Session')
    ax.set_ylabel('P(Left)')

    # plot block side
    ax.plot(np.arange(1, 1 + len(first_dt_of_each_trial)),
            np.where(first_dt_of_each_trial.block_side == -1, 0.8, 0.2),
            label='Block Side')

    # plot trial side
    ax.scatter(np.arange(1, 1 + len(first_dt_of_each_trial)),
               np.where(first_dt_of_each_trial.trial_side == -1, 1., 0.),
               label='Trial Side',
               alpha=0.8,
               s=1,
               c='tab:orange')
    ax.legend()
    hook_input['tensorboard_writer'].add_figure(
        tag='task_block_side_trial_side_by_trial_number',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_stimuli_by_block_side(hook_input):
    session_data = hook_input['session_data']

    fig, axes = plt.subplots(nrows=2, figsize=(4, 3))

    for i, block_side in enumerate(session_data.block_side.unique()):
        ax = axes[i]
        ax.set_title(f'{side_string_map[block_side]} Block')
        block_side_session_data = session_data[session_data.block_side == block_side]
        ax.hist(
            block_side_session_data.left_stimulus,
            bins=50,
            label='Left Stimulus',
            alpha=0.5,
            color=side_color_map['left'])
        ax.hist(
            block_side_session_data.right_stimulus,
            bins=50,
            label='Right Stimulus',
            alpha=0.5,
            color=side_color_map['right'])
        ax.legend()

    # add x label to lowest row
    ax.set_xlabel('Stimulus Value')
    hook_input['tensorboard_writer'].add_figure(
        tag='task_stimuli_by_block_side',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_stimuli_by_correct_trial_side(hook_input):
    session_data = hook_input['session_data']
    correct_side_stimuli = pd.concat(
        (session_data.right_stimulus[session_data.trial_side == 1],
         session_data.left_stimulus[session_data.trial_side == -1]),
        axis=0)
    incorrect_side_stimuli = pd.concat(
        (session_data.right_stimulus[session_data.trial_side == -1],
         session_data.left_stimulus[session_data.trial_side == 1]),
        axis=0)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Sampled Stimuli')
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    ax.scatter(
        correct_side_stimuli,
        incorrect_side_stimuli,
        alpha=0.5,
        color=side_color_map['neutral'])
    ax.axvline(np.mean(correct_side_stimuli.values))
    ax.axhline(np.mean(incorrect_side_stimuli.values))
    ax.set_xlabel('Correct Side Stimuli')
    ax.set_ylabel('Incorrect Side Stimuli')
    hook_input['tensorboard_writer'].add_figure(
        tag='task_stimuli_by_correct_trial_side',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def hook_plot_task_stimuli_and_model_prob_in_first_n_trials(hook_input):
    session_data = hook_input['session_data']

    ncols = 5
    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(4, 3),
        sharex=True,
        sharey=True)
    fig.text(0, 0, hook_input['model'].description_str, transform=fig.transFigure)
    axes[0].set_ylabel('Stimulus Strength & Model P(Left Trial)')

    for col, (_, trial_data) in enumerate(session_data.groupby(
            ['session_index', 'block_index', 'trial_index'])):

        if col == ncols:
            break

        ax = axes[col]
        ax.axhline(0.1, color='k')
        ax.axhline(0.9, color='k')
        ax.set_xlabel('RNN Step In Trial')
        ax.set_title(f'Trial Side: {side_string_map[trial_data.trial_side.unique()[0]]}\n'
                     f'Strength: {round(trial_data.trial_strength.unique()[0], 2)}')
        ax.set_xlim(0, hook_input['envs'][0].max_rnn_steps_per_trial)
        ax.plot(
            trial_data.rnn_step_index + 1,
            trial_data.left_stimulus,
            '+',  # necessary to ensure 1-RNN step trials visualized
            label='Left Stimulus',
            color=side_color_map['left'],
            markersize=6)
        ax.plot(
            trial_data.rnn_step_index + 1,
            trial_data.right_stimulus,
            '+',  # necessary to ensure 1-RNN step trials visualized
            label='Right Stimulus',
            color=side_color_map['right'],
            markersize=3)
        ax.plot(
            trial_data.rnn_step_index + 1,
            trial_data.left_action_prob,
            'o--',  # necessary to ensure 1-RNN step trials visualized
            label='Model P(Left Action)',
            color=side_color_map['left'],
            markersize=3)
    ax.legend()

    # add x label to lowest col
    hook_input['tensorboard_writer'].add_figure(
        tag='task_stimuli_and_model_prob_in_first_n_trials',
        figure=fig,
        global_step=hook_input['grad_step'],
        close=True if hook_input['tag_prefix'] != 'analyze/' else False)


def add_pca_readout_vectors_to_axis(ax, hook_input):
    # add readout vectors for right trial, right block
    labels = [
        '',
        '',
        # 'Right Trial Readout',
        # 'Right Block Readout'
    ]
    vectors = [hook_input['pca_trial_readout_vector'],
               hook_input['pca_block_readout_vector']
               ]
    colors = [
        side_color_map['neutral'],
        side_color_map['neutral']
    ]
    for i, (label, vector, color) in enumerate(zip(labels, vectors, colors)):
        # ax.arrow(x=0.,
        #          y=0.,
        #          dx=2 * vector[0],
        #          dy=2 * vector[1],
        #          color=color,
        #          length_includes_head=True,
        #          head_width=0.16,
        #          zorder=1)  # plot on top

        # calculate perpendicular hyperplane
        hyperplane = np.matmul(rotation_matrix_90, vector)
        np.testing.assert_almost_equal(actual=np.dot(hyperplane, vector),
                                       desired=0.)
        # scale hyperplane to ensure it covers entire plot
        hyperplane = 10 * hyperplane / np.linalg.norm(hyperplane)
        ax.plot([-hyperplane[0], 0, hyperplane[0]],
                [-hyperplane[1], 0, hyperplane[1]],
                color,
                zorder=1,  # plot on top
                dashes=[2, 2])

        ax.annotate(
            label,
            xy=(vector[0],
                vector[1]))
