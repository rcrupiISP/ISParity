# General imports
import time

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Fairlearn metrics
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               demographic_parity_ratio,
                               equalized_odds_difference, equalized_odds_ratio,
                               false_negative_rate,
                               false_negative_rate_difference,
                               false_positive_rate,
                               false_positive_rate_difference, selection_rate,
                               true_positive_rate_difference)
# Fairlearn post-processing algorithms
from fairlearn.postprocessing import ThresholdOptimizer
# ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             balanced_accuracy_score, f1_score,
                             mutual_info_score, normalized_mutual_info_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from generate_dataset import create_synth
from post_processing_ppv_for import (apply_decision_rule,
                                     run_ppv_parity_and_for_parity)


def fit_models(X_train, X_ind_train, X_supp_train, y_train):
    # Fit classifier
    # alternatives: LogisticRegression, DecisionTreeClassifier
    model = RandomForestClassifier
    clf = model(random_state=0, max_depth=5).fit(X_train, y_train)
    clf_ind = model(random_state=0, max_depth=5).fit(X_ind_train, y_train)
    clf_supp = model(random_state=0, max_depth=5).fit(X_supp_train, y_train)

    # KNN Individual metric
    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh.fit(X_ind_train, y_train)
    neigh_supp = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh_supp.fit(X_supp_train, y_train)

    return clf, clf_ind, clf_supp, neigh, neigh_supp


# code modified from https://github.com/joebaumann/fair-prediction-based-decision-making
def group_selection_rate(y, y_pred, group_indices):
    group_indices = group_indices.astype(bool)
    return sum(y_pred[group_indices] == 1) / len(y_pred[group_indices])


def tpr(y, y_pred, group_indices):
    group_indices = group_indices.astype(bool)
    if sum(y[group_indices] == 1) == 0:
        return 1.0
    return sum((y[group_indices] == 1) & (y_pred[group_indices] == 1)) / sum(y[group_indices] == 1)


def fpr(y, y_pred, group_indices):
    group_indices = group_indices.astype(bool)
    if sum(y[group_indices] == 0) == 0:
        return 1.0
    return sum((y[group_indices] == 0) & (y_pred[group_indices] == 1)) / sum(y[group_indices] == 0)

def ppv(y, y_pred, group_indices):
    group_indices = group_indices.astype(bool)
    if sum(y_pred[group_indices] == 1) == 0:
        return 1.0
    return sum((y[group_indices] == 1) & (y_pred[group_indices] == 1)) / sum(y_pred[group_indices] == 1)


def forate(y, y_pred, group_indices):
    group_indices = group_indices.astype(bool)
    if sum(y_pred[group_indices] == 0) == 0:
        return 0.0
    return sum((y[group_indices] == 1) & (y_pred[group_indices] == 0)) / sum(y_pred[group_indices] == 0)


def positive_predictive_value_difference(y_true, y_pred, sensitive_features):
    fairness_value_a1 = ppv(y_true, y_pred, sensitive_features)
    fairness_value_a0 = ppv(y_true, y_pred, 1-sensitive_features)
    return abs(fairness_value_a1-fairness_value_a0)


def false_omission_rate_difference(y_true, y_pred, sensitive_features):
    fairness_value_a1 = forate(y_true, y_pred, sensitive_features)
    fairness_value_a0 = forate(y_true, y_pred, 1-sensitive_features)
    return abs(fairness_value_a1-fairness_value_a0)


def sufficiency_difference(y_true, y_pred, sensitive_features):
    """Calculate the sufficiency difference.
    The greater of two metrics: `positive_predictive_value_difference` and
    `false_omission_rate_difference`. The former is the difference between the
    largest and smallest of :math:`P[Y=1 | A=a, D=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[Y=1 | A=a, Y=0]`.
    The sufficiency difference of 0 means that all groups have the same
    positive predictive value, false discovery rate, false omission rate, and negative predictive value.
    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.
    sensitive_features :
        The sensitive features over which demographic parity should be assessed
    Returns
    -------
    float
        The sufficiency difference
    """
    ppv_diff = positive_predictive_value_difference(
        y_true, y_pred, sensitive_features)
    for_diff = false_omission_rate_difference(
        y_true, y_pred, sensitive_features)
    return max(ppv_diff, for_diff)


def positive_predictive_value_ratio(y_true, y_pred, sensitive_features):
    fairness_value_a1 = ppv(y_true, y_pred, sensitive_features)
    fairness_value_a0 = ppv(y_true, y_pred, 1-sensitive_features)
    if max(fairness_value_a1, fairness_value_a0) == 0:
        return 0
    return min(fairness_value_a1, fairness_value_a0) / max(fairness_value_a1, fairness_value_a0)


def false_omission_rate_ratio(y_true, y_pred, sensitive_features):
    fairness_value_a1 = forate(y_true, y_pred, sensitive_features)
    fairness_value_a0 = forate(y_true, y_pred, 1-sensitive_features)
    if max(fairness_value_a1, fairness_value_a0) == 0:
        return 0
    return min(fairness_value_a1, fairness_value_a0) / max(fairness_value_a1, fairness_value_a0)


def sufficiency_ratio(y_true, y_pred, sensitive_features):
    """Calculate the sufficiency ratio.
    The smaller of two metrics: `positive_predictive_value_ratio` and
    `false_omission_rate_ratio`. The former is the ratio between the
    smallest and largest of :math:`P[Y=1 | A=a, D=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[Y=1 | A=a, D=0]`.
    The sufficiency ratio of 1 means that all groups have the same
    positive predictive value, false discovery rate, false omission rate, and negative predictive value.
    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.
    sensitive_features :
        The sensitive features over which demographic parity should be assessed
    Returns
    -------
    float
        The sufficiency ratio
    """
    ppv_ratio = positive_predictive_value_ratio(
        y_true, y_pred, sensitive_features)
    for_ratio = false_omission_rate_ratio(y_true, y_pred, sensitive_features)
    return min(ppv_ratio, for_ratio)

# Code modified from https://github.com/fairlearn/fairlearn/blob/main/notebooks/Binary%20Classification%20with%20the%20UCI%20Credit-card%20Default%20Dataset.ipynb
def get_metrics_df(models_dict, y_true, group, X_ind_test=None, X_supp_test=None, dct_flip=None):
    metrics_dict = {
        "Overall selection rate": (
            lambda x: selection_rate(y_true, x), True),
        "DP difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        "DP ratio": (
            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        # "Overall balanced error rate": (
        #     lambda x: 1-balanced_accuracy_score(y_true, x), True),
        # "Balanced error rate difference": (
        #     lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        "TPR difference": (
            lambda x: true_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "FPR difference": (
            lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "Equalized odds ratio": (
            lambda x: equalized_odds_ratio(y_true, x, sensitive_features=group), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "PPV difference": (
            lambda x: positive_predictive_value_difference(y_true, x, sensitive_features=group), True),
        "FOR difference": (
            lambda x: false_omission_rate_difference(y_true, x, sensitive_features=group), True),
        "Sufficiency ratio": (
            lambda x: sufficiency_ratio(y_true, x, sensitive_features=group), True),
        "Sufficiency difference": (
            lambda x: sufficiency_difference(y_true, x, sensitive_features=group), True),
        "Selection rate A0": (
            lambda x: group_selection_rate(y_true, x, group_indices=1-group), True),
        "Selection rate A1": (
            lambda x: group_selection_rate(y_true, x, group_indices=group), True),
        "TPR A0": (
            lambda x: tpr(y_true, x, group_indices=1-group), True),
        "TPR A1": (
            lambda x: tpr(y_true, x, group_indices=group), True),
        "FPR A0": (
            lambda x: fpr(y_true, x, group_indices=1-group), True),
        "FPR A1": (
            lambda x: fpr(y_true, x, group_indices=group), True),
        "PPV A0": (
            lambda x: ppv(y_true, x, group_indices=1-group), True),
        "PPV A1": (
            lambda x: ppv(y_true, x, group_indices=group), True),
        "FOR A0": (
            lambda x: forate(y_true, x, group_indices=1-group), True),
        "FOR A1": (
            lambda x: forate(y_true, x, group_indices=group), True),
        # "Overall AUC": (
        #     lambda x: roc_auc_score(y_true, x), False),
        "ACC score": (
            lambda x: accuracy_score(y_true, 1*(x > 0.5)), False),
        "F1 score": (
            lambda x: f1_score(y_true, 1*(x > 0.5)), False),
        "Precision": (
            lambda x: precision_score(y_true, 1*(x > 0.5)), False),
        "Recall": (
            lambda x: recall_score(y_true, 1*(x > 0.5)), False),
        # "AUC difference": (
        #     lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False),
        "ACC score difference": (
            lambda x: MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=1*(x > 0.5), sensitive_features=group).difference(method='between_groups'), False),
        "F1 score difference": (
            lambda x: MetricFrame(metrics=f1_score, y_true=y_true, y_pred=1*(x > 0.5), sensitive_features=group).difference(method='between_groups'), False),
        "Flip": (lambda x: dct_flip[x], 'no'),
        # "Individuality": (lambda x: neigh.score(X_ind_test, x), True),
        # "Individuality suppression": (lambda x: neigh_supp.score(X_supp_test, x), True),
        "Mutual information A, y_pred": (lambda x: normalized_mutual_info_score(group, x), True),
        "Mutual information A, y_true": (lambda x: normalized_mutual_info_score(group, y_true), True),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        list_tmp = []
        for model_name, (preds, scores) in models_dict.items():
            try:
                if use_preds == True:
                    list_tmp.append(metric_func(preds))
                elif use_preds == False:
                    list_tmp.append(metric_func(scores))
                else:
                    list_tmp.append(metric_func(model_name))
            except:
                print('problem for model:', model_name,
                      'in metric:', metric_name)
                list_tmp.append('NaN')
        df_dict[metric_name] = list_tmp
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())


def threshold_optimizer_ppv_for(s_train, y_train, group_indices, group_indices_test, s_test):
    threshold_nr = 100
    group_indices = [(1-group_indices).astype(
        bool), group_indices.astype(bool)]
    optimal_decision_rules_ppv, optimal_decision_rules_for = run_ppv_parity_and_for_parity(
        threshold_nr, s_train, y_train, group_indices)
    threshold_ppv_A0, threshold_ppv_A1 = optimal_decision_rules_ppv['ideal_thresholds']
    threshold_for_A0, threshold_for_A1 = optimal_decision_rules_for['ideal_thresholds']

    group_indices_test_A1 = group_indices_test.astype(bool)
    group_indices_test_A0 = (1-group_indices_test).astype(bool)

    postprocess_preds_ppv = apply_decision_rule(
        s_test, group_indices_test_A0, group_indices_test_A1, threshold_ppv_A0, threshold_ppv_A1)
    postprocess_preds_for = apply_decision_rule(
        s_test, group_indices_test_A0, group_indices_test_A1, threshold_for_A0, threshold_for_A1)

    return postprocess_preds_ppv, postprocess_preds_for


def mitigations(X_train, X_test, X_ind_test, X_supp_test, y_train, y_test, y_test_real,
                thr_supp, sens_var, cond_var,
                clf, clf_ind, clf_supp
                ):
    """Method to mitigate in postprocessing the predictions of the models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set
    X_test : pd.DataFrame
        Test set
    X_ind_test : pd.DataFrame
        Individual test set (without A)
    X_supp_test : pd.DataFrame
        Suppression test set (without varible correlated with A)
    y_train : pd.Series
        Target train
    y_test : pd.Series
        Target test
    y_test_real : pd.Series
        Target test - no measurement bias
    thr_supp: float
      Threshold for suppression method
    sens_var : str
        Name of the sensitive variable. E.g. sens_var = 'A'
    cond_var : str
        Name of the variable to condition used by the Conditional Demographic Parity.
        E.g. cond_var = 'Q'
    clf: sklearn.model
        Classifier model
    clf_ind: sklearn.model
        Classifier model for individual dataset
    clf_supp: sklearn.model
        Classifier model for suppression dataset

    Returns
    -------
    pd.DataFrame
        a DataFrame of the output results: metrics per each model.
    """

    # Define a dataset with sensitive/protected attribute flipped
    X_flip = X_test.copy()
    X_flip[sens_var] = 1-X_flip[sens_var]
    dct_flip = {'FTU': 1, 'Suppression_'+str(thr_supp): 1}

    # # # unmitigated
    dct_flip['Unmitigated'] = 1 - \
        abs(clf.predict(X_test) - clf.predict(X_flip)).mean()

    # # # TPR parity (=equality of opportunity)
    postprocess_est_tpr = ThresholdOptimizer(
        estimator=clf, constraints="true_positive_rate_parity", predict_method='predict_proba', prefit=True)
    postprocess_est_tpr.fit(
        X_train, y_train, sensitive_features=X_train[sens_var])
    postprocess_preds_tpr = postprocess_est_tpr.predict(
        X_test, sensitive_features=X_test[sens_var])
    postprocess_preds_tpr_flip = postprocess_est_tpr.predict(
        X_flip, sensitive_features=X_flip[sens_var])
    dct_flip['TPR'] = 1-abs(postprocess_preds_tpr -
                            postprocess_preds_tpr_flip).mean()

    # # # FPR parity
    postprocess_est_fpr = ThresholdOptimizer(
        estimator=clf, constraints="false_positive_rate_parity", predict_method='predict_proba', prefit=True)
    postprocess_est_fpr.fit(
        X_train, y_train, sensitive_features=X_train[sens_var])
    postprocess_preds_fpr = postprocess_est_fpr.predict(
        X_test, sensitive_features=X_test[sens_var])
    postprocess_preds_fpr_flip = postprocess_est_fpr.predict(
        X_flip, sensitive_features=X_flip[sens_var])
    dct_flip['FPR'] = 1-abs(postprocess_preds_fpr -
                            postprocess_preds_fpr_flip).mean()

    # # # PPV parity & FOR parity
    unmitigated_scores_train = pd.Series(
        clf.predict_proba(X_train)[:, 1], index=X_train.index)
    unmitigated_scores_test = pd.Series(
        clf.predict_proba(X_test)[:, 1], index=X_test.index)
    postprocess_preds_ppv, postprocess_preds_for = threshold_optimizer_ppv_for(
        unmitigated_scores_train, y_train, X_train[sens_var], X_test[sens_var], unmitigated_scores_test)

    # # # equalized_odds
    postprocess_est_eo = ThresholdOptimizer(estimator=clf,
                                            constraints="equalized_odds", predict_method='predict_proba', prefit=True)
    postprocess_est_eo.fit(
        X_train, y_train, sensitive_features=X_train[sens_var])
    postprocess_preds_eo = postprocess_est_eo.predict(
        X_test, sensitive_features=X_test[sens_var])
    postprocess_preds_eo_flip = postprocess_est_eo.predict(
        X_flip, sensitive_features=X_flip[sens_var])
    dct_flip['Separation'] = 1-abs(postprocess_preds_eo -
                           postprocess_preds_eo_flip).mean()

    # # # demographic_parity
    postprocess_est_dp = ThresholdOptimizer(estimator=clf,
                                            constraints="demographic_parity", predict_method='predict_proba', prefit=True)
    postprocess_est_dp.fit(
        X_train, y_train, sensitive_features=X_train[sens_var])
    postprocess_preds_dp = postprocess_est_dp.predict(
        X_test, sensitive_features=X_test[sens_var])
    postprocess_preds_dp_flip = postprocess_est_dp.predict(
        X_flip, sensitive_features=X_flip[sens_var])
    dct_flip['DP'] = 1-abs(postprocess_preds_dp -
                           postprocess_preds_dp_flip).mean()

    # # # Conditional demographic_parity
    try:
        postprocess_preds_cdp = 0*postprocess_preds_dp.copy()
        postprocess_preds_cdp_flip = 0*postprocess_preds_dp.copy()
        for i in X_train[cond_var].unique():
            postprocess_est_dp_0 = ThresholdOptimizer(estimator=clf,
                                                      constraints="demographic_parity", predict_method='predict_proba', prefit=True)
            postprocess_est_dp_0.fit(X_train.loc[X_train[cond_var] == i], y_train[X_train[cond_var] == i],
                                     sensitive_features=X_train.loc[X_train[cond_var] == i, sens_var])
            postprocess_preds_dp_0 = postprocess_est_dp_0.predict(X_test.loc[X_test[cond_var] == i],
                                                                  sensitive_features=X_test.loc[X_test[cond_var] == i, sens_var])
            postprocess_preds_dp_0_flip = postprocess_est_dp_0.predict(X_flip.loc[X_flip[cond_var] == i],
                                                                       sensitive_features=X_flip.loc[X_flip[cond_var] == i, sens_var])
            # Define the prediction with the same dimensionality of test prediction
            postprocess_preds_cdp[X_test[cond_var]
                                  == i] = postprocess_preds_dp_0
            postprocess_preds_cdp_flip[X_test[cond_var]
                                       == i] = postprocess_preds_dp_0_flip
        dct_flip['CDP'] = 1-abs(postprocess_preds_cdp -
                                postprocess_preds_cdp_flip).mean()
    except:
        print('Can t mitigate with CDP.')
        postprocess_preds_cdp = np.nan
        dct_flip['CDP'] = np.nan

    # # # Build dictionary of mitigated prediction
    models_dict = {"Y": (y_test, y_test),
                   "Y True": (y_test_real, y_test_real),
                   "Unmitigated": (clf.predict(X_test), clf.predict_proba(X_test)[:, 1]),
                   "FTU":  (clf_ind.predict(X_ind_test), clf_ind.predict_proba(X_ind_test)[:, 1]),
                   "Suppression_"+str(thr_supp): (clf_supp.predict(X_supp_test), clf_supp.predict_proba(X_supp_test)[:, 1]),
                   "Separation": (postprocess_preds_eo, postprocess_preds_eo),
                   "TPR": (postprocess_preds_tpr, postprocess_preds_tpr),
                   "FPR": (postprocess_preds_fpr, postprocess_preds_fpr),
                   "PPV": (postprocess_preds_ppv, postprocess_preds_ppv),
                   "FOR": (postprocess_preds_for, postprocess_preds_for),
                   "DP": (postprocess_preds_dp, postprocess_preds_dp),
                   "CDP": (postprocess_preds_cdp, postprocess_preds_cdp)}

    return models_dict, dct_flip


def pipeline(param_dict, sens_var='A', cond_var='Q', y_bias_meas=False):
    """Pipeline that create a synthetic dataset, fit the models, mitigate those
     and output the results metrics

    Parameters
    ----------
    param_dict : dict
        Dictonary for setting the dataset creation
    sens_var : str, optional
        Name of the sensitive variable
    cond_var : str, optional
        Name of the variable to condition used by the Conditional Demographic Parity
    y_bias_meas: bool, optional
        If true the metrics are tested on the target y without measurement bias

    Returns
    -------
    pd.DataFrame
        a DataFrame of the output results: metrics per each model.
    """
    print("The parameters for data generetion are: ", param_dict, '\n')
    thr_supp = param_dict["thr_supp"]
    # Create dataset
    print("Start creation dataset.", '\n')
    X_train, X_ind_train, X_supp_train, X_test, X_ind_test, X_supp_test, y_train, y_test, y_train_real, y_test_real = create_synth(
        **param_dict)
    df_total = X_train.copy()
    df_total['Y'] = y_train
    if y_bias_meas:
        df_total['Y_real'] = y_train_real
    print("The correlation matrix is: ", '\n', df_total.corr(), '\n',
          "The value counts is: ", '\n', df_total[sens_var].value_counts(), '\n')
    # Fit models
    print("Fitting models.", '\n')
    clf, clf_ind, clf_supp, neigh, neigh_supp = fit_models(
        X_train, X_ind_train, X_supp_train, y_train)
    # Mitigate models
    print("Mitigate models.", '\n')
    models_dict, dct_flip = \
        mitigations(X_train, X_test, X_ind_test, X_supp_test, y_train, y_test, y_test_real,
                    thr_supp, sens_var, cond_var,
                    clf, clf_ind, clf_supp)
    # return summary table
    print("Report output.", '\n')
    if y_bias_meas:
        return get_metrics_df(models_dict, y_test_real, X_test[sens_var], X_ind_test, X_supp_test, dct_flip)
    else:
        return get_metrics_df(models_dict, y_test, X_test[sens_var], X_ind_test, X_supp_test, dct_flip)


def timer(func):
    """Decorator that prints the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(
            f"\n\nDONE :)  --  Finished {func.__name__!r} in {run_time:.4f} seconds")
        return value

    return wrapper_timer

def set_plot_style():
    sns.set_context("paper")
    sns.set(font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })