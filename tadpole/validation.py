import numpy as np


def get_test_subjects(LB_table):
    """Load test subjects in LB2 cohort

    :param LB_table: Feature matrix
    :type LB_table: pd.DataFrame
    :return: Unique list of patient identifiers
    :rtype: np.ndarray
    """
    # * Copy the column specifying membership of LB2 into an array.
    LB2_inds = LB_table['RID'][LB_table.LB2 == 1]

    # * Get the list of subjects to forecast from LB2 - the ordering is the
    # * same as in the submission template.
    return np.unique(LB2_inds)
