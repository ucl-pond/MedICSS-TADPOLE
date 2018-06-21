import pandas as pd


def create_prediction(train_data, train_targets, data_forecast):
    """Create a simple prediction that just repeats the latest value.

    :param train_data: Features in training data.
    :type train_data: pd.DataFrame
    :param train_targets: Target in trainign data.
    :param pd.DataFrame
    :param data_forecast: Empty data to insert predictions into
    :type data_forecast: pd.DataFrame
    :return: Data frame in same format as data_forecast.
    :rtype: pd.DataFrame
    """
    # * Clinical status forecast: predefined likelihoods per current status
    most_recent_data = pd.concat((train_targets, train_data['EXAMDATE']), axis=1).sort_values(by='EXAMDATE')

    most_recent_CLIN_STAT = most_recent_data['CLIN_STAT'].dropna().tail(1).iloc[0]
    if most_recent_CLIN_STAT == 'NL':
        CNp, MCIp, ADp = 0.3, 0.4, 0.3
    elif most_recent_CLIN_STAT == 'MCI':
        CNp, MCIp, ADp = 0.1, 0.5, 0.4
    elif most_recent_CLIN_STAT == 'Dementia':
        CNp, MCIp, ADp = 0.15, 0.15, 0.7
    else:
        CNp, MCIp, ADp = 0.33, 0.33, 0.34

    # Use the same clinical status probabilities for all months
    data_forecast.loc[:, 'CN relative probability'] = CNp
    data_forecast.loc[:, 'MCI relative probability'] = MCIp
    data_forecast.loc[:, 'AD relative probability'] = ADp

    # * ADAS13 forecast: = most recent score, default confidence interval
    most_recent_ADAS13 = most_recent_data['ADAS13'].dropna().tail(1).iloc[0]
    data_forecast.loc[:, 'ADAS13'] = most_recent_ADAS13
    data_forecast.loc[:, 'ADAS13 50% CI lower'] = max([0, most_recent_ADAS13 - 1])
    data_forecast.loc[:, 'ADAS13 50% CI upper'] = most_recent_ADAS13 + 1

    # Subject has no history of ADAS13 measurement, so we'll take a
    # typical score of 12 with wide confidence interval +/-10.

    # * Ventricles volume forecast: = most recent measurement, default confidence interval
    most_recent_Ventricles_ICV = most_recent_data['Ventricles_ICV'].dropna().tail(1).iloc[0]
    data_forecast.loc[:, 'Ventricles_ICV'] = most_recent_Ventricles_ICV
    data_forecast.loc[:, 'Ventricles_ICV 50% CI lower'] = max(0, most_recent_Ventricles_ICV - 0.01 * most_recent_Ventricles_ICV)
    data_forecast.loc[:, 'Ventricles_ICV 50% CI upper'] = min(1, most_recent_Ventricles_ICV + 0.01 * most_recent_Ventricles_ICV)
    return data_forecast
