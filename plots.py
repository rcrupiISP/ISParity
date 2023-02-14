import math
from pathlib import Path

import colorcet as cc
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import (adjusted_mutual_info_score, mutual_info_score,
                             normalized_mutual_info_score)

from utils import *


def bias_mitigation_plot(detailed, results_df, bias_values, model_names, metrics, fig_title, xticks, xticklabels, xlabel, function, figsize):
    """This function generates a plot of the results obtained from the bias mitigation analysis. The plot shows how the value of a given metric varies with the value of the bias for each mitigation technique.

    Args:
        detailed (bool): whether to show detailed information about each metric or not.
        results_df (pandas DataFrame): the data containing the results of the bias mitigation analysis.
        bias_values (list): a list of values representing the magnitude of the bias for which the results were calculated.
        model_names (list): a list of names of the bias mitigation techniques whose results are being plotted.
        metrics (list): a list of metrics to be plotted.
        fig_title (str): the title of the plot.
        xticks (list): the values of the ticks on the x-axis.
        xticklabels (list): the labels of the ticks on the x-axis.
        xlabel (str): the label of the x-axis.
        function (function): the name of the bias that is investigated.
        figsize (tuple): the size of the plot.
    """
    # plot fig
    set_plot_style()
    ncols = 4
    nrows = math.ceil(len(model_names)/ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
    # delete subfigures if too many
    [fig.delaxes(axes[nrows-1][ncols-1-i])
     for i in range(ncols * nrows-len(model_names))]

    palette = iter(sns.color_palette(cc.glasbey_dark, 256))
    palette_barplot = iter(sns.color_palette(
        cc.glasbey_dark, 256, as_cmap=True))
    pal = {'Selection rate A0': (
        0.4980, 0.4980, 0.4980), 'Selection rate A1': (0.4980, 0.4980, 0.4980)}
    pal_barplot = {'Selection rate A0': '#808080',
                   'Selection rate A1': '#808080'}
    if detailed:
        # initialize color palette and markers
        markers = {'Selection rate A0': 'v', 'Selection rate A1': 'o'}
        for m in metrics:
            if m not in pal.keys():
                if m[-1] == '0':
                    m_other_group = m[:-1] + '1'
                elif m[-1] == '1':
                    m_other_group = m[:-1] + '0'
                if m_other_group in pal.keys():
                    pal[m] = pal[m_other_group]
                    pal_barplot[m] = pal_barplot[m_other_group]
                else:
                    pal[m] = next(palette)
                    pal_barplot[m] = next(palette_barplot)
                if m[-1] == '0':
                    markers[m] = 'v'
                elif m[-1] == '1':
                    markers[m] = 'o'
    else:
        # initialize color palette with selection rates being gray with markers
        for m in metrics:
            if m not in pal.keys():
                pal[m] = next(palette)
                pal_barplot[m] = next(palette_barplot)
        markers = {label: '' for label, color in pal.items()}
        selection_rate_markers = ['v', 'o']
        markers['Selection rate A0'] = selection_rate_markers[0]
        markers['Selection rate A1'] = selection_rate_markers[1]

    for model, ax in zip(model_names, axes.ravel()):
        if model == "Unmitigated":
            ax.set_title(f'{model}')
        else:
            ax.set_title(f'Mitigation: {model}')
        if len(bias_values) > 1:

            if detailed:
                sns.lineplot(ax=ax, data=results_df.loc[(results_df['model_name'] == model)], x="bias", y="value",
                             hue="metric", palette=pal, sort=False, style='metric', alpha=0.6, dashes=False, markers=markers)
            else:
                sns.lineplot(ax=ax, data=results_df.loc[(results_df['model_name'] == model) & (results_df['metric'] != "Selection rate A0") & (
                    results_df['metric'] != "Selection rate A1")], x="bias", y="value", hue="metric", palette=pal, sort=False, alpha=0.6)
                sns.lineplot(ax=ax, data=results_df.loc[(results_df['model_name'] == model) & ((results_df['metric'] == "Selection rate A0") | (
                    results_df['metric'] == "Selection rate A1"))], x="bias", y="value", hue="metric", palette=pal, style="metric", alpha=0.6, dashes=False, markers=selection_rate_markers)
            ax.get_legend().remove()
            if xlabel:
                plt.xlabel(xlabel)
                ax.set(xlabel=xlabel)
            # ax.set(ylabel=None)
        else:
            barplot_color = [pal_barplot[i]
                             for i in results_df.loc[results_df['model_name'] == model]['metric']]
            sns.barplot(ax=ax, data=results_df.loc[results_df['model_name'] ==
                        model], x="metric", y="value", palette=barplot_color, alpha=0.6)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if len(bias_values) > 1:
        # if fig_title == "Undersampling":
        #    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylim([0, 1])
        if bias_values[0] > bias_values[-1]:
            ax.invert_xaxis()
        if xticks:
            if xticklabels:
                plt.xticks(ticks=xticks, labels=xticklabels)
            else:
                plt.xticks(ticks=xticks)
        lines, labels = plt.gca().get_legend_handles_labels()
        # fig.supylabel('value')
        handles = [Line2D([0], [0], marker=markers[label], label=label,
                          color=color) for label, color in pal.items()]
        nr_of_legen_columns = 4
        plt.figlegend(handles=handles, loc='lower center',
                      ncol=nr_of_legen_columns, bbox_to_anchor=(0.5, -0.12), labelspacing=0.)
    plt.yticks(ticks=[0, 0.5, 1], labels=['0', '0.5', '1'])
    plt.tight_layout()
    base_path = f"plots/bias_plots"
    if detailed:
        base_path += '_detailed'
    # create directory if it does not exist yet
    Path(base_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{base_path}/{function.__name__}.pdf",
                bbox_inches='tight')
    plt.clf()


def get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values, fig_title, xticks=None, xticklabels=None, xlabel=None):
    """This decorator applies different mitigation techniques to datasets including various magnitudes of biases and plots the results w.r.t. performance and fairness metrics.

    Args:
        detailed (_type_): _description_
        mutual_information_plot (_type_): _description_
        bias_values (_type_): _description_
        fig_title (_type_): _description_
        xticks (_type_, optional): _description_. Defaults to None.
        xticklabels (_type_, optional): _description_. Defaults to None.
        xlabel (_type_, optional): _description_. Defaults to None.

        detailed (bool): if True, detailed fairness metrics for each group will used; if False, metrics w.r.t. group-level differences are used.
        mutual_information_plot (bool): if True, mutual information metrics will be plotted
        bias_values (list): a list of bias values (representing the magnitudes of bias) to be used in the analysis
        fig_title (str): title for the plot
        xticks (list or None): list of xtick locations for the plot
        xticklabels (list or None): list of xtick labels for the plot
        xlabel (str or None): label for the x-axis of the plot
    """
    def decorator(function):
        @timer
        def wrapper():
            print(bias_values, fig_title)
            print(
                f"\n###############\n     plotting {function.__name__} for the bias values {str(bias_values)} ...\n###############\n")
            results = []
            figsize, model_names = (9, 4), [
                'Unmitigated', 'DP', 'FTU', 'TPR', 'FPR', 'Separation', 'PPV', 'FOR']  # FULL results in appendix
            # figsize, model_names = (9, 3), ['Unmitigated', 'DP', 'FTU', 'TPR'] # part of the results in main text
            if detailed:
                metrics = ['Selection rate A0', 'Selection rate A1', 'TPR A0',
                           'TPR A1', 'FPR A0', 'FPR A1', 'PPV A0', 'PPV A1', 'FOR A0', 'FOR A1']
            else:
                metrics = ['Selection rate A0', 'Selection rate A1', 'ACC score', 'DP difference',
                           'TPR difference', 'FPR difference', 'PPV difference', 'FOR difference']
            for bias in bias_values:
                print(f"\nbias: {function.__name__} // value: {bias}\n")
                all_metrics = function(bias)
                if mutual_information_plot:
                    # append all mutual information metrics
                    results.extend(all_metrics)
                    continue
                for model in model_names:
                    for metric in metrics:
                        # append all accuracy and fairness metrics
                        results.append(
                            [bias, model, metric, all_metrics.loc[metric, model]])

            if mutual_information_plot:
                results_df = pd.DataFrame(results, columns=[
                                          'bias', 'col1', 'col2', 'col1_and_col2', 'mutual_info_metric', 'value'])
                return (fig_title, results_df)
            else:
                results_df = pd.DataFrame(
                    results, columns=['bias', 'model_name', 'metric', 'value'])
                bias_mitigation_plot(detailed, results_df, bias_values, model_names,
                                     metrics, fig_title, xticks, xticklabels, xlabel, function, figsize)
        return wrapper
    return decorator


def mutual_information(param_dict, mutual_info_metric, bias, sens_var='A', y_bias_meas=False):
    """Calculates the mutual information between variables in a synthetic dataset.
    """

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
    df_total['Y_pred'] = pd.Series(
        1*(clf.predict_proba(X_train)[:, 1] > 0.5), index=X_train.index)

    results = []
    # for col1 in df_total: # plot correlations between all variables
    for col1 in ['A']:  # use this for less messy mutual information plots to only show correlation between the sensitive attribute and other variables
        for col2 in df_total:
            # check if this combination already exists in list
            if col1 != col2 and len([True for _, c1, c2, _, _, _ in results if ((col1, col2) == (c1, c2) or (col1, col2) == (c2, c1))]) == 0:
                results.append([bias, col1, col2, f'{col1};{col2}', mutual_info_metric.__name__, mutual_info_metric(
                    df_total[col1], df_total[col2])])
    return results


def bias_plots(detailed=False, mutual_information_plot=False, mutual_info_metric='normalized_mutual_info_score'):
    dim = 100000

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0], fig_title="No bias")
    def no_bias(bias):
        # no bias
        # None
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Measurement bias on R", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^R_{m}$)")
    def measurement_bias_on_R(bias):
        # measurement on R
        # l_m = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": bias, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[False, True], fig_title="Omission bias", xticks=[0, 1], xticklabels=['False', 'True'])
    def omission_bias(bias):
        # omission
        # l_o = [False, True]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": bias, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00009], fig_title="Undersampling", xticks=[0.01, 0.005, 0.00009], xticklabels=['0.01', '0.005', '0.00009'], xlabel=r"bias ($p_u \perp\!\!\!\!\!\perp R$)")
    def undersampling(bias):
        # sample
        # p_u = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": bias, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Measurement bias on Y", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^Y_{m}$)")
    def measurement_bias_on_Y(bias):
        # measurement bias on Y (P_Y as target). Performance are calculated on Y
        # l_m_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": bias, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias, y_bias_meas=True)
        return pipeline(param_dict, y_bias_meas=True)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0005], fig_title="Representation bias", xticks=[1, 0.5, 0.0005], xticklabels=['1', '0.5', '0.0005'], xlabel=r"bias ($p_u \;\;\;\,\,\not\!\!\!\!\!\!\!\perp\!\!\!\!\!\perp R$)")
    def representation_bias(bias):
        # representation bias
        # p_u = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": bias, "l_r": True, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias on R", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^R_{h}$)")
    def historical_bias_on_R(bias):
        # historical bias on R
        # l_h_r = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": bias,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias on Y", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^Y_{h}$)")
    def historical_bias_on_Y(bias):
        # historical bias on Y
        # l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": bias, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias on Q", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^Q_{h}$)")
    def historical_bias_on_Q(bias):
        # historical bias on Q
        # l_h_q = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": bias,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias on R and Y", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^R_{h},\beta^Y_{h}$)")
    def historical_bias_on_R_and_Y(bias):
        # historical bias on R and Y
        # l_h_r and l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": bias, "l_m_y": 0, "thr_supp": 1, "l_h_r": bias,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias on Q and Y", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^Q_{h},\beta^Y_{h}$)")
    def historical_bias_on_Q_and_Y(bias):
        # historical bias on Q and Y
        # l_h_q and l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": bias, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0,  "l_h_q": bias,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias on Q and R and Y", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^Q_{h},\beta^R_{h},\beta^Y_{h}$)")
    def historical_bias_on_Q_and_R_and_Y(bias):
        # historical bias on Q and R and Y
        # l_h_r and l_h_q and l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": bias, "l_m_y": 0, "thr_supp": 1, "l_h_r": bias,  "l_h_q": bias,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias and measurement bias on R", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^R_{h},\beta^R_{m}$)")
    def historical_bias_and_measurement_bias_on_R(bias):
        # historical bias on R
        # l_h_r and l_m = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": bias,  "l_h_q": 0,
                      "l_m": bias, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    @get_metrics_for_various_bias_values(detailed, mutual_information_plot, bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], fig_title="Historical bias and measurement bias on Y", xticks=[0, 3, 6, 9], xlabel=r"bias ($\beta^Y_{h},\beta^Y_{m}$)")
    def historical_bias_and_measurement_bias_on_Y(bias):
        # historical bias and measurement bias on Y
        # l_y and l_m_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim": dim, "l_y": bias, "l_m_y": bias, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0, "l_q": 2, "sy": 2, "l_r_q": 0}
        if mutual_information_plot:
            return mutual_information(param_dict, mutual_info_metric, bias)
        return pipeline(param_dict)

    if mutual_information_plot:
        results = []
        # specify the scenarios for which the mutual information should be calculated
        results.append(no_bias())
        results.append(measurement_bias_on_R())
        results.append(measurement_bias_on_Y())
        results.append(historical_bias_on_R())
        results.append(historical_bias_on_Y())
        results.append(historical_bias_on_Q())
        results.append(undersampling())
        results.append(representation_bias())
        results.append(omission_bias())
        results.append(historical_bias_and_measurement_bias_on_R())
        results.append(historical_bias_and_measurement_bias_on_Y())
        return results

    else:
        # specify the scenarios to be investigated
        no_bias()
        measurement_bias_on_R()
        measurement_bias_on_Y()
        historical_bias_on_R()
        historical_bias_on_Y()
        historical_bias_on_Q()
        undersampling()
        representation_bias()
        omission_bias()
        historical_bias_and_measurement_bias_on_R()
        historical_bias_and_measurement_bias_on_Y()


def mutual_information_plot(mutual_info_metric=normalized_mutual_info_score):
    """Generates a mutual information plot using the given mutual information metric.

    Args:
        mutual_info_metric (function or string): The mutual information metric to use. If 'correlation', a correlation function is created to calculate mutual information. Defaults to normalized_mutual_info_score.
    """
    if mutual_info_metric == 'correlation':
        def mutual_info_metric(x, y): return np.corrcoef(x, y)[0, 1]
        mutual_info_metric.__name__ = 'correlation'

    mutual_information_metrics = bias_plots(
        mutual_information_plot=True, mutual_info_metric=mutual_info_metric)

    # plot fig
    set_plot_style()
    ncols = 4
    nrows = math.ceil(len(mutual_information_metrics)/ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=False, sharey=True, figsize=(10, 8))
    [fig.delaxes(axes[nrows-1][ncols-1-i]) for i in range(ncols*nrows -
                                                          len(mutual_information_metrics))]  # delete subfigures if too many
    fig.suptitle(f'Metric: {mutual_info_metric.__name__}')
    # plt.subplots_adjust(hspace=0.5)

    # set color for each possible combination of features and labels
    # max_nr_of_combinations = max([len(i[1]["col1_and_col2"].unique()) for i in mutual_information_metrics])
    palette = iter(sns.color_palette(cc.glasbey_dark, 256))
    pal = {}
    for _, results_df in mutual_information_metrics:
        for i in results_df['col1_and_col2']:
            if i not in pal.keys():
                pal[i] = next(palette)

    for (fig_title, results_df), ax in zip(mutual_information_metrics, axes.ravel()):
        ax.set_title(fig_title)
        if len(results_df['bias'].unique()) > 1:
            sns.lineplot(ax=ax, data=results_df, x="bias", y="value",
                         hue="col1_and_col2", palette=pal, sort=False, alpha=0.6)
            ax.get_legend().remove()
        else:
            sns.barplot(ax=ax, data=results_df, x="col1_and_col2",
                        y="value", palette=pal, alpha=0.6)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # ax.set(ylabel=None)
        # if fig_title == "Undersampling":
        #    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # ax.set_ylim([0, 1])
    # if bias_values[0] > bias_values[-1]:
    #    ax.invert_xaxis()
    # fig.supylabel('value')
    handles = [Line2D([0], [0], label=label, color=color)
               for label, color in pal.items()]
    plt.figlegend(handles=handles, loc='lower center', ncol=7,
                  bbox_to_anchor=(0.5, -0.1), labelspacing=0.)
    plt.tight_layout()
    base_path = f"plots/mutual_information_plots"
    # create directory if it does not exist yet
    Path(base_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{base_path}/{mutual_info_metric.__name__}.pdf", bbox_inches='tight')
    plt.clf()


def calibration_plot_meas_bias_y():
    """Generate calibration plot for measurement bias on Y (P_Y as target).
    """

    set_plot_style()
    palette = iter(sns.color_palette(cc.glasbey, 256))
    plt.rcParams["figure.figsize"] = (8, 4)

    def add_line_calibration_plot(param_dict, sens_var='A', cond_var='Q', y_bias_meas=False, plot_calib_counter=0, label=""):
        thr_supp = param_dict["thr_supp"]
        # Create dataset
        X_train, X_ind_train, X_supp_train, X_test, X_ind_test, X_supp_test, y_train, y_test, y_train_real, y_test_real = create_synth(
            **param_dict)
        df_total = X_train.copy()
        df_total['Y'] = y_train
        if y_bias_meas:
            df_total['Y_real'] = y_train_real
        # Fit models
        clf, clf_ind, clf_supp, neigh, neigh_supp = fit_models(
            X_train, X_ind_train, X_supp_train, y_train)

        # Plot calibration curve
        from sklearn.calibration import calibration_curve
        for group in [1]:
            if plot_calib_counter > 0 and group == 0:
                continue

            group_indices = X_test[sens_var] == group

            prob = clf.predict_proba(X_test[group_indices])

            if plot_calib_counter == 0:
                # Plot perfectly calibrated
                plt.plot([0, 1], [0, 1], linestyle='--', color="black",
                         alpha=0.5, label='Ideally Calibrated')
            # Plot model's calibration curve for real Y
            x, y = calibration_curve(
                y_test_real[group_indices], prob[:, 1], n_bins=10)
            plt.plot(y, x, marker='x', alpha=0.7, color=next(palette),
                     label=r'$Y$  '+label)

            # Plot model's calibration curve for biased Y
            x, y = calibration_curve(
                y_test[group_indices], prob[:, 1], n_bins=10)
            plt.plot(y, x, marker='.', alpha=0.7, color=next(palette),
                     label=r'$P_Y$ '+label)

            leg = plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.xlabel(f'Average predicted probability in each bin')
            plt.ylabel('Ratio of positives')
            # plt.show()

        return

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    # for plot_calib_counter,x in enumerate([0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]):
    for plot_calib_counter, x in enumerate([0, 1, 3, 6, 9]):
        print("nr", plot_calib_counter)
        label = r'[$\beta^Y_{m}$='+f'{x}]'
        param_dict = {"dim": 1000000, "l_y": 0, "l_m_y": x, "thr_supp": 1, "l_h_r": 0,  "l_h_q": 0,
                      "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                      "l_q": 2, "sy": 2, "l_r_q": 0}
        dtf_res = add_line_calibration_plot(
            param_dict, y_bias_meas=True, plot_calib_counter=plot_calib_counter, label=label)
    # create directory if it does not exist yet
    plt.tight_layout()
    base_path = f"plots/calibration_plots"
    # create directory if it does not exist yet
    Path(base_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"{base_path}/calibration_plots_measurement_bias_on_Y.pdf", bbox_inches='tight')


if __name__ == "__main__":
    # generate accuracy fairness plots for various biases and mitigation techniques
    bias_plots()  # plot difference metrics, e.g., TPR diff, PPV diff, ...

    # plot detailed group-specific metrics, e.g., TPR A0, TPR A1, PPV A0, PPV A1, ...
    bias_plots(detailed=True)

    # generate mutual information plot for various biases
    # possible metrics are mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score, 'correlation'
    mutual_information_plot('correlation')

    # generate plot for measurement bias on Y
    calibration_plot_meas_bias_y()
