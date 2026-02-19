import math
from scipy import stats


# -------------------------
# MEAN FUNCTIONS
# -------------------------

def arithmetic_mean(data):
    """
    Calculates the arithmetic mean of a dataset.
    """
    total = 0
    for x in data:
        total += x
    return total / len(data)


def harmonic_mean(data):
    """
    Calculates the harmonic mean of a dataset.
    Ignores zero values to avoid division by zero.
    """
    reciprocal_sum = 0
    count = 0

    for x in data:
        if x > 0:  # ignore zeros
            reciprocal_sum += 1 / x
            count += 1

    if count == 0:
        return 0

    return count / reciprocal_sum

# -------------------------
# STANDARD DEVIATION
# -------------------------

def standard_deviation(data):
    """
    Calculates the sample standard deviation of a dataset.
    """
    mean = arithmetic_mean(data)
    squared_diff_sum = 0

    for x in data:
        squared_diff_sum += (x - mean) ** 2

    variance = squared_diff_sum / len(data)
    return math.sqrt(variance)

    # -------------------------
# POOLED STANDARD DEVIATION
# -------------------------

def pooled_std(std_list, n_list):
    """
    Calculates pooled standard deviation
    given lists of standard deviations and sample sizes.
    """
    numerator = 0
    denominator = 0

    for s, n in zip(std_list, n_list):
        numerator += (n - 1) * (s ** 2)
        denominator += (n - 1)

    return math.sqrt(numerator / denominator)

# -------------------------
# INDEPENDENT T-TEST
# -------------------------

def t_test(data1=None, data2=None,
           mu1=None, mu2=None,
           sigma1=None, sigma2=None,
           n1=None, n2=None,
           mean_type="arithmetic"):
    """
    Performs independent samples t-test.
    Can accept raw datasets OR summary statistics.
    """

    # If raw data provided
    if data1 is not None and data2 is not None:

        if mean_type == "harmonic":
            mu1 = harmonic_mean(data1)
            mu2 = harmonic_mean(data2)
        else:
            mu1 = arithmetic_mean(data1)
            mu2 = arithmetic_mean(data2)

        sigma1 = standard_deviation(data1)
        sigma2 = standard_deviation(data2)
        n1 = len(data1)
        n2 = len(data2)

    df = n1 + n2 - 2

    sp = math.sqrt(
        ((n1 - 1) * sigma1**2 + (n2 - 1) * sigma2**2) / df
    )

    t_value = (mu1 - mu2) / (sp * math.sqrt((1/n1) + (1/n2)))

    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), df))

    return t_value, p_value

# -------------------------
# ONE-WAY ANOVA
# -------------------------

def one_way_anova(*groups):
    """
    Performs one-way ANOVA on 3 or more groups.
    """

    # Combine all data
    all_data = []
    for group in groups:
        all_data.extend(group)

    overall_mean = arithmetic_mean(all_data)

    ss_between = 0
    ss_within = 0

    for group in groups:
        group_mean = arithmetic_mean(group)
        n = len(group)

        # Between-group variation
        ss_between += n * (group_mean - overall_mean) ** 2

        # Within-group variation
        for x in group:
            ss_within += (x - group_mean) ** 2

    df_between = len(groups) - 1
    df_within = len(all_data) - len(groups)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F = ms_between / ms_within

    p_value = 1 - stats.f.cdf(F, df_between, df_within)

    return F, p_value

# -------------------------
# REPEATED MEASURES ANOVA
# -------------------------

def repeated_measures_anova(data_matrix):
    """
    Performs repeated measures ANOVA.
    data_matrix should be a list of lists:
    rows = subjects
    columns = conditions
    """

    n = len(data_matrix)          # number of subjects
    k = len(data_matrix[0])       # number of conditions

    # Flatten all values
    all_values = [x for row in data_matrix for x in row]
    overall_mean = arithmetic_mean(all_values)

    # Subject means
    subject_means = [arithmetic_mean(row) for row in data_matrix]

    # Condition means
    condition_means = []
    for j in range(k):
        col = [data_matrix[i][j] for i in range(n)]
        condition_means.append(arithmetic_mean(col))

    # SS_subjects
    ss_subjects = 0
    for i in range(n):
        for j in range(k):
            ss_subjects += (data_matrix[i][j] - subject_means[i]) ** 2

    # SS_conditions
    ss_conditions = 0
    for j in range(k):
        ss_conditions += n * (condition_means[j] - overall_mean) ** 2

    ss_error = ss_subjects - ss_conditions

    df_conditions = k - 1
    df_subjects = n - 1
    df_error = df_conditions * df_subjects

    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error

    F = ms_conditions / ms_error

    p_value = 1 - stats.f.cdf(F, df_conditions, df_error)

    return F, p_value
