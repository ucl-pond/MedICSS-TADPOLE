import pandas as pd
import numpy as np
import datetime as dt


def get_age_at_exam(data):
    """Compute data at exam from age at baseline and date of exam

    :param data: Input data
    :type data: pd.DataFrame
    :return: Data frame with AGE_AT_EXAM and AGE_INT
    :rtype: pd.DataFrame
    """
    def add_age(x):
        values = x["EXAMDATE"]
        val = (values - values.min()).dt.days / 365.25 + x["AGE"].min()
        return val

    tadpole_grouped = data.groupby("RID").apply(add_age)
    age_data = pd.DataFrame({
        'AGE_AT_EXAM': tadpole_grouped,
        'AGE_INT': tadpole_grouped.astype(int)})
    return age_data


def load_tadpole_data(filename):
    """Load input data of the TADPOLE challenge

    :param filename: Path to TADPOLE_LB1_LB2.csv file
    :return: The first element is a data frame of features,
    the second element a data frame of target variables.
    :rtype: tuple of pd.DataFrame
    """
    # * Read in the D1_D2 spreadsheet: may give a DtypeWarning, but the read/import works.
    LB_table = pd.read_csv(filename, low_memory=False)

    # * Target variables: convert strings to numeric if necessary
    target_variables = ['DX', 'ADAS13', 'Ventricles']
    variables_to_check = ['RID', 'ICV_bl'] + target_variables[1:]  # also check RosterID and IntraCranialVolume@
    for var_name in variables_to_check:
        var0 = LB_table[var_name].iloc[0]
        if isinstance(var0, str):
            # * Convert strings to numeric
            LB_table[var_name] = LB_table[var_name].astype(np.int)

    LB_table['Ventricles_ICV'] = LB_table['Ventricles'] / LB_table['ICV_bl']

    def last_diagnosis(x):
        if pd.notnull(x):
            return x.split(' ')[-1]

    LB_table['CLIN_STAT'] = LB_table['DX'].apply(last_diagnosis)

    # * Compute months since Jan 2000 for each exam date
    ref = dt.datetime(2000, 1, 1)
    LB_table['EXAMDATE'] = pd.to_datetime(LB_table.EXAMDATE)

    LB_table['ExamMonth'] = (LB_table['EXAMDATE'] - ref).dt.days / 365 * 12

    target_variables.extend(['Ventricles_ICV', 'CLIN_STAT'])
    y = LB_table.loc[:, ['RID'] + target_variables]
    X = LB_table.drop(target_variables, axis=1)

    age = get_age_at_exam(X)
    # need to match multi-index
    X_mindex = X.set_index('RID', append=True).swaplevel()
    X = pd.concat((X_mindex, age), axis=1).reset_index(level=0)

    return X, y


def write_submission_table(submission, output_file):
    """Write submission table to disk.

    :param submission: List of data frames with predictions
    :rtype: list of pd.DataFrame
    :param outputFile: Path to output file
    """
    submission_table = pd.concat(submission, axis=0)
    submission_table['Forecast Date'] = submission_table['Forecast Date'].apply(lambda d: d.strftime('%Y-%m'))
    submission_table.to_csv(output_file, index=False)
