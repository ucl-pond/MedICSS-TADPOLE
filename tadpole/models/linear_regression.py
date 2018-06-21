import pandas as pd
import numpy as np


def create_prediction(train_data, train_targets, data_forecast):
    """Create a linear regression prediction that does a first order
    extrapolation in time of ADAS13 and ventricles.

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
    most_recent_data = pd.concat((train_targets, train_data[['EXAMDATE', 'AGE_AT_EXAM']]), axis=1).sort_values(by='EXAMDATE')

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
    adas_mask = (most_recent_data['AGE_AT_EXAM'].dropna()>0) & (most_recent_data['ADAS13'].dropna()>0) # not missing: Ventricles and ICV
    x = most_recent_data['AGE_AT_EXAM'].dropna()[adas_mask]
    y = most_recent_data['ADAS13'].dropna()[adas_mask]
    lm = np.polyfit(x,y,1)
    adas_p = np.poly1d(lm)

    adas_prediction = np.maximum(most_recent_ADAS13, adas_p(most_recent_data['AGE_AT_EXAM'].dropna().iloc[-1] +
                                                data_forecast['Forecast Month']/12))
    data_forecast.loc[:, 'ADAS13'] = adas_prediction
    data_forecast.loc[:, 'ADAS13 50% CI lower'] = adas_prediction - 1
    data_forecast.loc[:, 'ADAS13 50% CI upper'] = adas_prediction + 1

    # Subject has no history of ADAS13 measurement, so we'll take a
    # typical score of 12 with wide confidence interval +/-10.

    # * Ventricles volume forecast: = most recent measurement, default confidence interval
    most_recent_Ventricles_ICV = most_recent_data['Ventricles_ICV'].dropna().tail(1).iloc[0]

    vent_mask = (most_recent_data['Ventricles_ICV'].dropna()>0) & (most_recent_data['AGE_AT_EXAM'].dropna()>0) # not missing: Ventricles and ICV
    x = most_recent_data['AGE_AT_EXAM'].dropna()[vent_mask]
    y = most_recent_data['Ventricles_ICV'].dropna()[vent_mask]
    lm = np.polyfit(x,y,1)
    vent_p = np.poly1d(lm)

    vent_prediction = vent_p(most_recent_data['AGE_AT_EXAM'].dropna().iloc[-1] + data_forecast['Forecast Month']/12)

    data_forecast.loc[:, 'Ventricles_ICV'] = vent_prediction
    data_forecast.loc[:, 'Ventricles_ICV 50% CI lower'] = np.maximum(0, vent_prediction - 0.01 * vent_prediction)
    data_forecast.loc[:, 'Ventricles_ICV 50% CI upper'] = np.minimum(1, vent_prediction + 0.01 * vent_prediction)
    return data_forecast
