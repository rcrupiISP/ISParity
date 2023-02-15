# code modified from https://github.com/joebaumann/fair-prediction-based-decision-making
from statistics import mean
import numpy as np
import pandas as pd
import sklearn.metrics as sk
from sympy import Symbol
from sympy.solvers import solve
import utils

seed = 42


class DisparityFunction:
    """
    The disparity function determines to what degree a certain fairness metric must be fullfilled.
    Since fairness criteria are usually defined as parity metrics, there are different possibilities to compare, e.g., two groups' TPR rates.
    This class implements three different possible types of disparity functions.
    Using the example of two groups' (a and b) TPR rates (TPR_a and TPR_b) and a given value x, these are defined as follows:
    - diff: the difference between the two groups' rates is measured
        formally: TPR_a - TPR_b = x
    - p_percentage: the minimum ratio between the two groups rates' must be equal to some value x
        formally: min(TPR_a/TPR_b , TPR_b/TPR_a) = x
    - p_percentage_LEQ: the minimum ratio between the two groups rates' must be larger or equal than some value x
        formally: min(TPR_a/TPR_b , TPR_b/TPR_a) >= x
    """

    def __init__(self, disparity_function_type, fairness_constraint_value, metric_values_utility, metric_values_fair, fairness_metric_name):
        self.disparity_function_type = disparity_function_type
        self.fairness_constraint_value = fairness_constraint_value
        self.metric_values_utility = metric_values_utility
        self.metric_values_fair = metric_values_fair
        self.fairness_metric_name = fairness_metric_name

    def diff(self, a, b):
        return a - b

    def p_percentage(self, a, b):
        if a == 0 or b == 0:
            return float('inf')
        return min((a/b), (b/a))

    def calculate_solutions(self, rates_a, rates_b, y, s, thresholds, group_indices, utility_function):
        utilities = []
        thresholds_a = []
        thresholds_b = []

        if self.disparity_function_type == "diff":
            disparity_function = self.diff
            max_disparity = disparity_function(
                *self.metric_values_utility[self.fairness_metric_name])
            min_disparity = disparity_function(
                *self.metric_values_fair[self.fairness_metric_name])
            target_disparity = min_disparity + \
                (1 - self.fairness_constraint_value) * \
                disparity_function(max_disparity, min_disparity)

            for rate_a, threshold_a in zip(rates_a, thresholds):
                rate_a = np.float64(rate_a)
                target_rate = disparity_function(rate_a, target_disparity)
                if target_rate >= 0 and target_rate <= 1:
                    threshold_b = thresholds[np.argmin(
                        np.abs(rates_b - target_rate))]
                    y_pred = y_pred_two_thresholds(
                        s, *group_indices, threshold_a, threshold_b)
                    utility = utility_function(y, s, y_pred)
                    utilities.append(utility)
                    thresholds_a.append(threshold_a)
                    thresholds_b.append(threshold_b)

        elif self.disparity_function_type == "p_percentage":
            disparity_function = self.p_percentage
            for rate_a, threshold_a in zip(rates_a, thresholds):
                rate_a = np.float64(rate_a)
                # get the threshold producing p_percentage that is closest to gamma
                threshold_b = thresholds[np.argmin([abs(disparity_function(
                    rate_a, i) - self.fairness_constraint_value) for i in rates_b])]
                y_pred = y_pred_two_thresholds(
                    s, *group_indices, threshold_a, threshold_b)
                utility = utility_function(y, s, y_pred)
                utilities.append(utility)
                thresholds_a.append(threshold_a)
                thresholds_b.append(threshold_b)

        elif self.disparity_function_type == "p_percentage_LEQ":
            disparity_function = self.p_percentage
            for rate_a, threshold_a in zip(rates_a, thresholds):
                rate_a = np.float64(rate_a)
                # get the threshold producing the max utility while producing a p_percentage that is >= gamma
                threshold_b = thresholds[np.argmax([utility_function(y, s, y_pred_two_thresholds(s, *group_indices, threshold_a, thresholds[enum])) if (
                    disparity_function(rate_a, rate) >= self.fairness_constraint_value) else -np.inf for enum, rate in enumerate(rates_b)])]
                y_pred = y_pred_two_thresholds(
                    s, *group_indices, threshold_a, threshold_b)
                utility = utility_function(y, s, y_pred)
                utilities.append(utility)
                thresholds_a.append(threshold_a)
                thresholds_b.append(threshold_b)

        return utilities, thresholds_a, thresholds_b


def find_highest_utility_under_fairness(y, s, thresholds, group_indices, fairness_function, utility_function, y_pred_dict=None):
    """
    This function derives optimal decision rules under a specific fairness constraint, which takes the form of a parity metric.
    """

    if y_pred_dict == None:
        y_pred_dict = {t: s.between(t[0], t[1]) for t in thresholds}

    rates_a = [fairness_function(
        y, y_pred_dict[t], group_indices[0]) for t in thresholds]
    rates_b = [fairness_function(
        y, y_pred_dict[t], group_indices[1]) for t in thresholds]

    utilities = []
    thresholds_a = []
    thresholds_b = []
    for rate_a, threshold_a in zip(rates_a, thresholds):
        rate_a = np.float64(rate_a)
        threshold_b = thresholds[np.argmin(np.abs(rates_b - rate_a))]
        y_pred = y_pred_two_thresholds(
            s, *group_indices, threshold_a, threshold_b)
        utility = utility_function(y, s, y_pred)
        utilities.append(utility)
        thresholds_a.append(threshold_a)
        thresholds_b.append(threshold_b)

    highest_utility_index = np.argmax(utilities)
    highest_utility = utilities[highest_utility_index]
    ideal_rate = rates_a[highest_utility_index]
    ideal_threshold_a = thresholds_a[highest_utility_index]
    ideal_threshold_b = thresholds_b[highest_utility_index]
    return rates_a, rates_b, highest_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b


def find_highest_utility(y, s, thresholds, group_indices, utility_function, y_pred_dict=None):
    """
    This function derives optimal decision rules from a set of possible treshold rules, without the consideration of fairness.
    """

    if y_pred_dict == None:
        y_pred_dict = {t: s.between(t[0], t[1]) for t in thresholds}

    utilities_a = [utility_function(
        y[group_indices[0]], s[group_indices[0]], y_pred_dict[t][group_indices[0]]) for t in thresholds]
    utilities_b = [utility_function(
        y[group_indices[1]], s[group_indices[1]], y_pred_dict[t][group_indices[1]]) for t in thresholds]

    highest_utility_index_a = np.argmax(utilities_a)
    highest_utility_index_b = np.argmax(utilities_b)
    ideal_threshold_a = thresholds[highest_utility_index_a]
    ideal_threshold_b = thresholds[highest_utility_index_b]
    y_pred = y_pred_two_thresholds(
        s, *group_indices, ideal_threshold_a, ideal_threshold_b)
    highest_utility = utility_function(y, s, y_pred)
    return utilities_a, utilities_b, highest_utility, ideal_threshold_a, ideal_threshold_b


def y_pred_two_thresholds(s, group_indices_a, group_indices_b, thresholds_a, thresholds_b):
    y_pred = pd.Series([False] * len(s), index=s.index)
    if type(thresholds_a) is tuple:
        y_pred[group_indices_a] = s[group_indices_a].between(
            thresholds_a[0], thresholds_a[1])
        y_pred[group_indices_b] = s[group_indices_b].between(
            thresholds_b[0], thresholds_b[1])
    else:
        y_pred[group_indices_a] = s[group_indices_a] >= thresholds_a
        y_pred[group_indices_b] = s[group_indices_b] >= thresholds_b
    y_pred = y_pred.astype(bool)
    return y_pred


def calculate_metrics(metrics, y, s, y_pred_dict, ideal_threshold_a, ideal_threshold_b, group_indices):
    metric_values = {}
    for metric_name, fairness_function in metrics.items():
        fairness_value_a = fairness_function(
            y, y_pred_dict[ideal_threshold_a], group_indices[0])
        fairness_value_b = fairness_function(
            y, y_pred_dict[ideal_threshold_b], group_indices[1])
        metric_values[metric_name] = [fairness_value_a, fairness_value_b]

    return metric_values


def generate_thresholds(threshold_nr):
    thresholds = []
    for i in np.linspace(1/threshold_nr, 1, threshold_nr).tolist():
        thresholds.append((0, i))
    for i in np.linspace(1/threshold_nr, 1-1/threshold_nr, threshold_nr-1).tolist():
        thresholds.append((i, 1.0))
    return thresholds


def generate_upper_and_lower_bound_thresholds(s, nr_of_thresholds, probable_unconstrained_optimal_threshold):
    """
    This function produces thresholds based on the underlying score distribution, so that between every 2 thresholds there are the same amount of individuals.
    """

    results, bin_edges = pd.qcut(
        s, q=nr_of_thresholds, retbins=True, duplicates="drop")
    thresholds = list(bin_edges)
    if len(s) < nr_of_thresholds:
        print("nr_of_thresholds is set to", nr_of_thresholds, "but sample size is just", len(
            s), ". This resulted in duplicate thresholds. Non-unique thresholds were droped, which resulted in a total of", len(thresholds), "thresholds.")

    if min(s) < 0.1 and min(s) >= 0:
        thresholds.insert(0, 0.0)
    if max(s) > 0.9 and max(s) <= 1:
        thresholds.append(1.0)

    threshold_tuples = []

    minimum, maximum = min(thresholds), max(thresholds)

    if probable_unconstrained_optimal_threshold > maximum or probable_unconstrained_optimal_threshold < minimum:
        print("Error: probable_unconstrained_optimal_threshold must be between the minimum and the maximum threshold.")
        return

    for t in thresholds:
        new_tuple = (minimum, t)
        if minimum != t and new_tuple not in threshold_tuples:
            threshold_tuples.append(new_tuple)
    for t in thresholds:
        new_tuple = (t, maximum)
        if maximum != t and new_tuple not in threshold_tuples:
            threshold_tuples.append(new_tuple)

    # insert threshold of optimal unconstrained classifier
    if (probable_unconstrained_optimal_threshold, maximum) not in threshold_tuples and (float(probable_unconstrained_optimal_threshold), maximum) not in threshold_tuples:
        # check lower-bound thresholds
        for i, t in enumerate(threshold_tuples):
            if t[0] > probable_unconstrained_optimal_threshold:
                threshold_tuples.insert(
                    i, (float(probable_unconstrained_optimal_threshold), maximum))
                break
        if (probable_unconstrained_optimal_threshold, maximum) not in threshold_tuples and (float(probable_unconstrained_optimal_threshold), maximum) not in threshold_tuples:
            threshold_tuples.append(
                (float(probable_unconstrained_optimal_threshold), maximum))

    if (minimum, probable_unconstrained_optimal_threshold) not in threshold_tuples and (minimum, float(probable_unconstrained_optimal_threshold)) not in threshold_tuples:
        # check upper-bound thresholds
        for i, t in enumerate(threshold_tuples):
            if probable_unconstrained_optimal_threshold > t[1]:
                threshold_tuples.insert(
                    i-1, (minimum, float(probable_unconstrained_optimal_threshold)))
                break
        if (minimum, probable_unconstrained_optimal_threshold) not in threshold_tuples and (minimum, float(probable_unconstrained_optimal_threshold)) not in threshold_tuples:
            threshold_tuples.insert(
                0, (minimum, float(probable_unconstrained_optimal_threshold)))

    return threshold_tuples


class UtilityFunction:

    def __init__(self, u_tn, u_fp, u_fn, u_tp, proba, normalize=True):
        self.u_tn = u_tn
        self.u_fp = u_fp
        self.u_fn = u_fn
        self.u_tp = u_tp
        if proba:
            self.calculate_utility = self.calculate_utility_proba
            self.get_utility_with_randomization = self.get_utility_with_randomization_proba
        else:
            self.calculate_utility = self.calculate_utility_not_proba
            self.get_utility_with_randomization = self.get_utility_with_randomization_not_proba
        self.normalize = normalize

    def get_optimal_unconstrained_decision_rule(self):
        # this function calculates the p for which the decision maker's utility is 0
        return (self.u_tn - self.u_fp) / (self.u_tp - self.u_fn - self.u_fp + self.u_tn)

    def calculate_utility_proba(self, y, s, y_pred):
        D_eq_0_probabilities = s[np.invert(y_pred)]
        D_eq_1_probabilities = s[y_pred]

        D_eq_0_probabilities_avg = 0 if (
            len(D_eq_0_probabilities) <= 0) else mean(D_eq_0_probabilities)
        D_eq_1_probabilities_avg = 0 if (
            len(D_eq_1_probabilities) <= 0) else mean(D_eq_1_probabilities)

        D_eq_0_utility_avg = self.u_fn * D_eq_0_probabilities_avg + \
            self.u_tn * (1 - D_eq_0_probabilities_avg)
        D_eq_1_utility_avg = self.u_tp * D_eq_1_probabilities_avg + \
            self.u_fp * (1 - D_eq_1_probabilities_avg)

        total_utility = len(D_eq_0_probabilities) * D_eq_0_utility_avg + \
            len(D_eq_1_probabilities) * D_eq_1_utility_avg
        return total_utility

    def calculate_utility_not_proba(self, y_true, s, y_pred, sample_weight=None):
        y_type, y_true, y_pred = sk._classification._check_targets(
            y_true, y_pred)
        sk._classification.check_consistent_length(
            y_true, y_pred, sample_weight)
        if y_type.startswith('multilabel'):
            raise ValueError("Classification metrics can't handle multilabels")
            # differing_labels = sk._classification.count_nonzero(y_true - y_pred, axis=1)
            # score = differing_labels == 0
        else:
            tn, fp, fn, tp = sk.confusion_matrix(
                y_true=y_true, y_pred=y_pred).ravel()
            score = self.u_tn * tn + self.u_fp * fp + self.u_fn * fn + self.u_tp * tp

        if self.normalize:
            score = score/len(y_pred)
        return score

    def get_utility_with_randomization_not_proba(self, s, y_pred_dict, group_indices, target):
        # TODO: implement the function to derive a randomized decision rule with non-probabilistic fairness measures
        raise Exception("This function is not yet implemented. :(")

    def get_utility_with_randomization_proba(self, s, y_pred_dict, group_indices, target):
        D_eq_0_probabilities = s[group_indices][np.invert(y_pred_dict)[
            group_indices]]
        D_eq_1_probabilities = s[group_indices][y_pred_dict[group_indices]]

        D_eq_0_probabilities_avg = 0 if (
            len(D_eq_0_probabilities) <= 0) else mean(D_eq_0_probabilities)
        D_eq_1_probabilities_avg = 0 if (
            len(D_eq_1_probabilities) <= 0) else mean(D_eq_1_probabilities)

        D_eq_0_nr_of_individuals = len(D_eq_0_probabilities)
        D_eq_1_nr_of_individuals = len(D_eq_1_probabilities)

        # deviate from optimum: FOR rate a = base rate b
        x = Symbol("x")
        nr_of_individuals_to_switch_from_d1_to_d0 = solve(
            ((D_eq_0_probabilities_avg * D_eq_0_nr_of_individuals + D_eq_1_probabilities_avg * x) / (D_eq_0_nr_of_individuals + x) - target), x)[0]

        randomization = nr_of_individuals_to_switch_from_d1_to_d0 / \
            D_eq_1_nr_of_individuals

        D_eq_0_nr_of_individuals += nr_of_individuals_to_switch_from_d1_to_d0
        D_eq_1_nr_of_individuals -= nr_of_individuals_to_switch_from_d1_to_d0

        D_eq_0_utility_avg = self.u_fn * target + \
            self.u_tn * (1 - D_eq_0_probabilities_avg)
        D_eq_1_utility_avg = self.u_tp * D_eq_1_probabilities_avg + \
            self.u_fp * (1 - D_eq_1_probabilities_avg)

        total_utility = D_eq_0_nr_of_individuals * D_eq_0_utility_avg + \
            D_eq_1_nr_of_individuals * D_eq_1_utility_avg
        return total_utility, randomization


def get_PPV_FOR_rates_for_all_decision_rules(y, s, thresholds, group_indices, fairness_functions, utility_function, y_pred_dict=None):

    if y_pred_dict == None:
        y_pred_dict = {t: s.between(t[0], t[1]) for t in thresholds}
    rates_a_PPV = np.array([fairness_functions[0](
        y, s, y_pred_dict[t], group_indices[0]) for t in thresholds])
    rates_a_FOR = np.array([fairness_functions[1](
        y, s, y_pred_dict[t], group_indices[0]) for t in thresholds])
    rates_b_PPV = np.array([fairness_functions[0](
        y, s, y_pred_dict[t], group_indices[1]) for t in thresholds])
    rates_b_FOR = np.array([fairness_functions[1](
        y, s, y_pred_dict[t], group_indices[1]) for t in thresholds])

    return rates_a_PPV, rates_a_FOR, rates_b_PPV, rates_b_FOR


def generate_solution_space_and_maximize_utility(y, group0_base_rate, group1_base_rate, rates_a_PPV, rates_a_FOR, rates_b_PPV, rates_b_FOR, thresholds, group_indices, my_utility_function, optimal_unconstrained_decision_rule, s, y_pred_dict):
    max_base_rate = max(group0_base_rate, group1_base_rate)
    min_base_rate = min(group0_base_rate, group1_base_rate)

    solution_space = {}
    for i in zip(rates_a_PPV, rates_a_FOR, thresholds):
        rate_a_PPV = i[0]
        rate_a_FOR = i[1]
        t = i[2]
        if (not (0 <= rate_a_PPV <= 1)) or (not (0 <= rate_a_FOR <= 1)):
            min_FOR = None
            max_FOR = None
            index_of_closest_PPV_rate = None
            rate_b_PPV = None
            rate_b_FOR = None
            threshold_b = None
        else:
            index_of_closest_PPV_rate = np.argmin(
                np.abs(rates_b_PPV - rate_a_PPV))
            rate_b_PPV = rates_b_PPV[index_of_closest_PPV_rate]
            rate_b_FOR = rates_b_FOR[index_of_closest_PPV_rate]
            threshold_b = thresholds[index_of_closest_PPV_rate]

            if np.abs(rate_b_PPV - rate_a_PPV) > 0.05:
                min_FOR = None
                max_FOR = None

            elif rate_a_FOR <= rate_b_FOR:
                min_FOR = rate_a_FOR
                max_FOR = rate_b_FOR
            else:
                min_FOR = rate_b_FOR
                max_FOR = rate_a_FOR

        solution_space[t] = {"rate_a_PPV": rate_a_PPV, "rate_a_FOR": rate_a_FOR, "threshold_b": threshold_b,
                             "rate_b_index": index_of_closest_PPV_rate, "rate_b_PPV": rate_b_PPV, "rate_b_FOR": rate_b_FOR, "min_FOR": min_FOR, "max_FOR": max_FOR}

    # get thresholds for which a solution exists
    solution_space_without_None_values = {k: v for (k, v) in solution_space.items(
    ) if (v["min_FOR"] != None and v["max_FOR"] != None)}

    min_FOR = np.array(list({k: v["min_FOR"] for (
        k, v) in solution_space_without_None_values.items()}.values()))
    max_FOR = np.array(list({k: v["max_FOR"] for (
        k, v) in solution_space_without_None_values.items()}.values()))

    rates_a_PPV_without_None_values = np.array(list({k: v["rate_a_PPV"] for (
        k, v) in solution_space_without_None_values.items()}.values()))

    remaining_thresholds = list(
        {k: v for (k, v) in solution_space_without_None_values.items()}.keys())
    previous_t = None
    for t in remaining_thresholds:
        if t[0] != 0:
            turning_point_t_index = remaining_thresholds.index(previous_t)
            break
        previous_t = t

    def get_utility_maximizing_ppv_for_combination(solution_space_without_None_values, y, group0_base_rate, group1_base_rate, group_indices, my_utility_function, optimal_unconstrained_decision_rule, s, y_pred_dict):
        # go through all edge cases of solution space to find the one with maximal utility
        optimal_solution = {}

        for threshold_a, v in solution_space_without_None_values.items():
            optimal_solution_for_this_PPV = {}
            if v["rate_a_PPV"] <= optimal_unconstrained_decision_rule:
                # we maximize utility by minimizing the number of individuals with D=1
                if v["rate_a_PPV"] <= group0_base_rate and v["rate_a_PPV"] <= group1_base_rate:
                    # we minimize the number of individuals with D=1 by minimizing FOR_rate

                    if v["rate_a_FOR"] < max(group0_base_rate, group1_base_rate) or v["rate_b_FOR"] < max(group0_base_rate, group1_base_rate):
                        # no solution exists
                        threshold_a, randomization_a, utility_a = None, None, None
                        v["threshold_b"], randomization_b, utility_b = None, None, None
                        FOR_rate = None
                    elif group0_base_rate <= group1_base_rate:
                        FOR_rate = group1_base_rate

                        utility_a, randomization_a = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[threshold_a], group_indices[0], target=group1_base_rate)  # deviate from optimum: FOR rate a = base rate b

                        v["threshold_b"] = (0, 1)
                        randomization_b = 0
                        utility_b = my_utility_function.calculate_utility(y[group_indices[1]], s[group_indices[1]], np.invert(
                            y_pred_dict[v["threshold_b"]])[group_indices[1]])
                    else:
                        FOR_rate = group0_base_rate
                        threshold_a = (0, 1)
                        randomization_a = 0
                        utility_a = my_utility_function.calculate_utility(y[group_indices[0]], s[group_indices[0]], np.invert(
                            y_pred_dict[threshold_a])[group_indices[0]])

                        utility_b, randomization_b = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[v["threshold_b"]], group_indices[1], target=group0_base_rate)  # deviate from optimum: FOR rate b = base rate a

                elif v["rate_a_PPV"] >= group0_base_rate and v["rate_a_PPV"] >= group1_base_rate:
                    # we minimize the number of individuals with D=1 by maximizing FOR_rate

                    if v["rate_a_FOR"] > min(group0_base_rate, group1_base_rate) or v["rate_b_FOR"] > min(group0_base_rate, group1_base_rate):
                        # no solution exists
                        threshold_a, randomization_a, utility_a = None, None, None
                        v["threshold_b"], randomization_b, utility_b = None, None, None
                        FOR_rate = None
                    elif group0_base_rate <= group1_base_rate:
                        FOR_rate = group0_base_rate

                        threshold_a = (0, 1)
                        randomization_a = 0
                        utility_a = my_utility_function.calculate_utility(y[group_indices[0]], s[group_indices[0]], np.invert(
                            y_pred_dict[threshold_a])[group_indices[0]])

                        utility_b, randomization_b = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[v["threshold_b"]], group_indices[1], target=group0_base_rate)  # deviate from optimum: FOR rate b = base rate a
                    else:
                        FOR_rate = group1_base_rate
                        utility_a, randomization_a = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[threshold_a], group_indices[0], target=group1_base_rate)  # deviate from optimum: FOR rate a = base rate b

                        v["threshold_b"] = (0, 1)
                        randomization_b = 0
                        utility_b = my_utility_function.calculate_utility(y[group_indices[1]], s[group_indices[1]], np.invert(
                            y_pred_dict[v["threshold_b"]])[group_indices[1]])

                else:
                    # no solution exists
                    threshold_a, randomization_a, utility_a = None, None, None
                    v["threshold_b"], randomization_b, utility_b = None, None, None
                    FOR_rate = None

            else:
                # we maximize utility by maximizing the number of individuals with D=1
                if v["rate_a_PPV"] <= group0_base_rate and v["rate_a_PPV"] <= group1_base_rate:
                    # we maximize the number of individuals with D=1 by maximizing FOR_rate

                    if v["rate_a_FOR"] < max(group0_base_rate, group1_base_rate) or v["rate_b_FOR"] < max(group0_base_rate, group1_base_rate):
                        # no solution exists
                        threshold_a, randomization_a, utility_a = None, None, None
                        v["threshold_b"], randomization_b, utility_b = None, None, None
                        FOR_rate = None
                    elif v["rate_a_FOR"] <= v["rate_b_FOR"]:
                        # rate_a_FOR is smaller so this is given
                        FOR_rate = v["rate_a_FOR"]
                        randomization_a = 1
                        utility_a = my_utility_function.calculate_utility(
                            y[group_indices[0]], s[group_indices[0]], y_pred_dict[threshold_a][group_indices[0]])

                        utility_b, randomization_b = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[v["threshold_b"]], group_indices[1], target=v["rate_a_FOR"])  # deviate from optimum: FOR rate b = FOR rate a
                    else:
                        # rate_b_FOR is smaller so this is given
                        FOR_rate = v["rate_b_FOR"]
                        utility_a, randomization_a = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[threshold_a], group_indices[0], target=v["rate_b_FOR"])  # deviate from optimum: FOR rate a = FOR rate b

                        randomization_b = 1
                        utility_b = my_utility_function.calculate_utility(
                            y[group_indices[1]], s[group_indices[1]], y_pred_dict[v["threshold_b"]][group_indices[1]])

                elif v["rate_a_PPV"] >= group0_base_rate and v["rate_a_PPV"] >= group1_base_rate:
                    # we maximize the number of individuals with D=1 by minimizing FOR_rate

                    if v["rate_a_FOR"] > min(group0_base_rate, group1_base_rate) or v["rate_b_FOR"] > min(group0_base_rate, group1_base_rate):
                        # no solution exists
                        threshold_a, randomization_a, utility_a = None, None, None
                        v["threshold_b"], randomization_b, utility_b = None, None, None
                        FOR_rate = None
                    elif v["rate_a_FOR"] <= v["rate_b_FOR"]:
                        FOR_rate = v["rate_b_FOR"]
                        utility_a, randomization_a = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[threshold_a], group_indices[0], target=v["rate_b_FOR"])  # deviate from optimum: FOR rate a = FOR rate b

                        randomization_b = 1
                        utility_b = my_utility_function.calculate_utility(
                            y[group_indices[1]], s[group_indices[1]], y_pred_dict[v["threshold_b"]][group_indices[1]])
                    else:
                        FOR_rate = v["rate_a_FOR"]
                        randomization_a = 1
                        utility_a = my_utility_function.calculate_utility(
                            y[group_indices[1]], s[group_indices[0]], y_pred_dict[threshold_a][group_indices[0]])

                        utility_b, randomization_b = my_utility_function.get_utility_with_randomization(
                            s, y_pred_dict[threshold_a], group_indices[1], target=v["rate_a_FOR"])  # deviate from optimum: FOR rate b = FOR rate a

                else:
                    # no solution exists
                    threshold_a, randomization_a, utility_a = None, None, None
                    threshold_b, randomization_b, utility_b = None, None, None
                    FOR_rate = None

            optimal_solution_for_this_PPV["threshold_a"] = threshold_a
            optimal_solution_for_this_PPV["threshold_b"] = v["threshold_b"]
            optimal_solution_for_this_PPV["randomization_a"] = randomization_a
            optimal_solution_for_this_PPV["randomization_b"] = randomization_b
            optimal_solution_for_this_PPV["utility_a"] = utility_a
            optimal_solution_for_this_PPV["utility_b"] = utility_b
            optimal_solution_for_this_PPV["PPV_rate"] = v["rate_a_PPV"]
            optimal_solution_for_this_PPV["FOR_rate"] = FOR_rate
            if utility_a != None and utility_b != None:
                optimal_solution_for_this_PPV["total_utility"] = utility_a + utility_b
            else:
                optimal_solution_for_this_PPV["total_utility"] = np.NINF

            # only add if solution exists
            if threshold_a != None:
                optimal_solution[threshold_a] = optimal_solution_for_this_PPV

        return optimal_solution

    optimal_solution = get_utility_maximizing_ppv_for_combination(
        solution_space_without_None_values, y, group0_base_rate, group1_base_rate, group_indices, my_utility_function, optimal_unconstrained_decision_rule, s, y_pred_dict)

    return optimal_solution, max_base_rate, min_base_rate, solution_space_without_None_values, rates_a_PPV_without_None_values, turning_point_t_index, min_FOR, max_FOR


def run_ppv_parity_and_for_parity(threshold_nr, s, y, group_indices):
    my_utility_function = UtilityFunction(
        1, 0, 0, 1, proba=False)  # corresponds to accuracy
    thresholds = generate_upper_and_lower_bound_thresholds(
        s, threshold_nr, 0.5)
    y_pred_dict = {t: s.between(t[0], t[1]) for t in thresholds}

    """
    # Maximize utility

    utilities_a, utilities_b, highest_utility, ideal_threshold_a, ideal_threshold_b = find_highest_utility(
        y=y, s=s, thresholds=thresholds, group_indices=group_indices, utility_function=my_utility_function.calculate_utility)

    metric_values_utility = calculate_metrics({"ppv": utils.ppv, "for": utils.forate}, y, s, y_pred_dict, ideal_threshold_a, ideal_threshold_b, group_indices)

    print("\nIf we maximize utility without fairness constraints, these are the optimal thresholds:",
          "\n  ideal_threshold_A0:", ideal_threshold_a, "\n  ideal_threshold_A1:", ideal_threshold_b)
    print("Thereby, we achieve the following fairness values:", metric_values_utility)

    no_fairness_PPV_rate_0, no_fairness_FOR_rate_0 = metric_values_utility["ppv"][0], metric_values_utility["for"][0]
    no_fairness_PPV_rate_1, no_fairness_FOR_rate_1 = metric_values_utility["ppv"][1], metric_values_utility["for"][1]
    group_selection_rate_0, group_selection_rate_1 = metric_values_utility["group_selection_rate"][0], metric_values_utility["group_selection_rate"][1]

    optimal_decision_rules_unconstrained = {
        "ideal_thresholds": (ideal_threshold_a, ideal_threshold_b),
        "no_fairness_PPV A0": no_fairness_PPV_rate_0,
        "no_fairness_FOR A0": no_fairness_FOR_rate_0,
        "no_fairness_PPV A1": no_fairness_PPV_rate_1,
        "no_fairness_FOR A1": no_fairness_FOR_rate_1,
        "group_selection_rate_A0": group_selection_rate_0,
        "group_selection_rate_A1": group_selection_rate_1
    }
    """

    # Find ideal thresholds under PPV parity

    fairness_function = utils.ppv

    rates_a, rates_b, highest_fair_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b = find_highest_utility_under_fairness(
        y=y, s=s, thresholds=thresholds, group_indices=group_indices, fairness_function=fairness_function, utility_function=my_utility_function.calculate_utility, y_pred_dict=y_pred_dict)

    FOR_rates_a = [utils.forate(
        y, y_pred_dict[t], group_indices[0]) for t in thresholds]
    FOR_rates_b = [utils.forate(
        y, y_pred_dict[t], group_indices[1]) for t in thresholds]

    group_selection_rates_a = [utils.group_selection_rate(
        y, y_pred_dict[t], group_indices[0]) for t in thresholds]
    group_selection_rates_b = [utils.group_selection_rate(
        y, y_pred_dict[t], group_indices[1]) for t in thresholds]

    print("\nUnder PPV parity, these are the optimal thresholds:", "\n  ideal_threshold_A0:", ideal_threshold_a, "\n  ideal_threshold_A1:", ideal_threshold_b, "\n  PPV rates: ", ideal_rate,
          "\nThis results in the following FOR rates:", "\n  FOR rate group 0:", FOR_rates_a[thresholds.index(ideal_threshold_a)], "\n  FOR rate group 1:", FOR_rates_b[thresholds.index(ideal_threshold_b)])

    PPV_parity_ideal_PPV_rate = ideal_rate
    PPV_parity_ideal_FOR_rate_0 = FOR_rates_a[thresholds.index(
        ideal_threshold_a)]
    PPV_parity_ideal_FOR_rate_1 = FOR_rates_b[thresholds.index(
        ideal_threshold_b)]

    group_selection_rate_0 = group_selection_rates_a[thresholds.index(ideal_threshold_a)]
    group_selection_rate_1 = group_selection_rates_b[thresholds.index(ideal_threshold_b)]

    optimal_decision_rules_ppv = {
        "ideal_thresholds": (ideal_threshold_a, ideal_threshold_b),
        "PPV_parity_ideal_PPV_rate": PPV_parity_ideal_PPV_rate,
        "PPV_parity_ideal_FOR A0": PPV_parity_ideal_FOR_rate_0,
        "PPV_parity_ideal_FOR A1": PPV_parity_ideal_FOR_rate_1,
        "group_selection_rate_A0": group_selection_rate_0,
        "group_selection_rate_A1": group_selection_rate_1
    }

    # Find ideal thresholds for FOR parity

    fairness_function = utils.forate

    rates_a, rates_b, highest_fair_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b = find_highest_utility_under_fairness(
        y=y, s=s, thresholds=thresholds, group_indices=group_indices, fairness_function=fairness_function, utility_function=my_utility_function.calculate_utility, y_pred_dict=y_pred_dict)

    PPV_rates_a = [utils.ppv(y, y_pred_dict[t], group_indices[0])
                   for t in thresholds]
    PPV_rates_b = [utils.ppv(y, y_pred_dict[t], group_indices[1])
                   for t in thresholds]

    group_selection_rates_a = [utils.group_selection_rate(
        y, y_pred_dict[t], group_indices[0]) for t in thresholds]
    group_selection_rates_b = [utils.group_selection_rate(
        y, y_pred_dict[t], group_indices[1]) for t in thresholds]

    print("\nUnder FOR parity, these are the optimal thresholds:", "\n  ideal_threshold_A0:", ideal_threshold_a, "\n  ideal_threshold_A1:", ideal_threshold_b, "\n  FOR rates: ", ideal_rate,
          ".\nThis results in the following PPV rates:", "\n  PPV rate group 0:", PPV_rates_a[thresholds.index(ideal_threshold_a)], "\n  PPV rate group 1:", PPV_rates_b[thresholds.index(ideal_threshold_b)])

    FOR_parity_ideal_FOR_rate = ideal_rate
    FOR_parity_ideal_PPV_rate_0 = PPV_rates_a[thresholds.index(
        ideal_threshold_a)]
    FOR_parity_ideal_PPV_rate_1 = PPV_rates_b[thresholds.index(
        ideal_threshold_b)]

    group_selection_rate_0 = group_selection_rates_a[thresholds.index(ideal_threshold_a)]
    group_selection_rate_1 = group_selection_rates_b[thresholds.index(ideal_threshold_b)]

    optimal_decision_rules_for = {
        "ideal_thresholds": (ideal_threshold_a, ideal_threshold_b),
        "FOR_parity_ideal_FOR_rate": FOR_parity_ideal_FOR_rate,
        "FOR_parity_ideal_PPV A0": FOR_parity_ideal_PPV_rate_0,
        "FOR_parity_ideal_PPV A1": FOR_parity_ideal_PPV_rate_1,
        "group_selection_rate_A0": group_selection_rate_0,
        "group_selection_rate_A1": group_selection_rate_1
    }

    return optimal_decision_rules_ppv, optimal_decision_rules_for


def apply_decision_rule(s, group_indices_a, group_indices_b, thresholds_a, thresholds_b):
    y_pred = pd.Series([False] * len(s), index=s.index)
    if type(thresholds_a) is tuple:
        y_pred[group_indices_a] = s[group_indices_a].between(
            thresholds_a[0], thresholds_a[1])
        y_pred[group_indices_b] = s[group_indices_b].between(
            thresholds_b[0], thresholds_b[1])
    else:
        y_pred[group_indices_a] = s[group_indices_a] >= thresholds_a
        y_pred[group_indices_b] = s[group_indices_b] >= thresholds_b
    y_pred = y_pred.astype(bool)
    return y_pred
