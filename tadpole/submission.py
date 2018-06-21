import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta


def create_submission_table(subjects, n_forecasts):
    """Create empty data frame in submission format.

    :param subjects: List of subject IDs
    :type subjects: list|tuple
    :param n_forecasts: Number of time points (months) to foreacast
    :type n_forecasts: int
    :return: Empty data frame
    :rtype: pd.DataFrame
    """
    # As opposed to the actual submission, we require 84 months of forecast
    # data. This is because some ADNI2 subjects in LB4 have visits as long as
    # 7 years after their last ADNI1 visit in LB2

    columns = ['RID',
               'Forecast Month',
               'Forecast Date',
               # * 1. Clinical status
               'CN relative probability',
               'MCI relative probability',
               'AD relative probability',
               # * 2. ADAS13 score
               'ADAS13',
               'ADAS13 50% CI lower',
               'ADAS13 50% CI upper',
               # * 3. Ventricles volume (normalised by intracranial volume)
               'Ventricles_ICV',
               'Ventricles_ICV 50% CI lower',
               'Ventricles_ICV 50% CI upper',
               ]

    subjects = np.asarray(subjects)
    # * Repeated matrices - compare with submission template
    rid = pd.Series(subjects.repeat(n_forecasts), name='RID')
    n_subjects = len(subjects)

    submission_table = pd.DataFrame(columns=columns, index=rid.index)
    submission_table['RID'] = rid
    submission_table['Forecast Month'] = np.tile(range(1, n_forecasts + 1), (n_subjects, 1)).flatten()

    # * Submission dates - compare with submission template
    forecast_dates = [dt.datetime(2010, 5, 1) + relativedelta(months=i) for i in range(n_forecasts)]

    # forecast_dates_strings = [dt.datetime.strftime(d, '%Y-%m') for d in forecast_dates]
    submission_table['Forecast Date'] = np.tile(forecast_dates, (n_subjects, 1)).flatten()

    return submission_table
