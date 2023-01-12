from pathlib import Path
from utils import *
import seaborn as sns
import colorcet as cc
import functools


def bias_and_group_fairness_for_various_bias_values(bias_values):
    def decorator(function):
        @timer
        def wrapper():
            print(f"\n###############\n     plotting {function.__name__} for the bias values {str(bias_values)} ...\n###############\n")
            results = []
            model_names = ['Unmitigated', 'TPR', 'FPR', 'PPV', 'FOR', 'EO', 'DP', 'FTU']
            #metrics = ['ACC score', 'Demographic parity difference', 'TPR difference', 'FPR difference', 'Equalized odds difference', 'PPV difference', 'FOR difference', 'Sufficiency difference']
            metrics = ['ACC score', 'Demographic parity difference', 'TPR difference', 'FPR difference', 'PPV difference', 'FOR difference']
            for bias in bias_values:
                print(f"\nbias: {function.__name__} // value: {bias}\n")
                all_metrics = function(bias)
                for model in model_names:
                    for metric in metrics:
                        if metric == 'Demographic parity difference':
                            metric_label = 'DP difference'
                        else:
                            metric_label = metric
                        results.append([bias, model, metric_label, all_metrics.loc[metric, model]])
            results_df = pd.DataFrame(results, columns=['bias_value', 'model_name', 'metric', 'value'])
            # plot fig
            for model in model_names:
                if len(bias_values) > 2:
                    ax = sns.lineplot(data=results_df.loc[results_df['model_name'] == model], x="bias_value", y="value", hue="metric", sort=False, alpha=0.6, legend='full')
                    ax.set_ylim([0, 1])
                    if bias_values[0]>bias_values[-1]:
                        ax.invert_xaxis()
                elif len(bias_values) > 1:
                    ax = sns.barplot(data=results_df.loc[results_df['model_name'] == model], x="metric", y="value", hue="bias_value", alpha=0.6)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                else:
                    ax = sns.barplot(data=results_df.loc[results_df['model_name'] == model], x="metric", y="value", alpha=0.6)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                base_path = f"plots/bias_and_group_fairness/{model}"
                # create directory if it does not exist yet
                Path(base_path).mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{base_path}/{function.__name__}.pdf", bbox_inches='tight')
                plt.clf()
        return wrapper
    return decorator

def bias_and_group_fairness():
    dim = 100000
    #dim = 10000

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0])
    def no_bias(bias):
        # no bias
        # None
        param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":0, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 1, 2, 4, 8, 9, 10])
    def measurement_on_R(bias):
        # measurement on R
        # l_m = [0, 1, 2, 4, 8, 9, 10]
        param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":0, "l_m":bias, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[False, True])
    def omission(bias):
        # omission
        # l_o = [False, True]
        param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":0, "l_m":0, "p_u":1, "l_r":False, "l_o":bias, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    #@bias_and_group_fairness_for_various_bias_values(bias_values=[1, 0.5, 0.1, 0.05, 0.01])
    def undersampling(bias):
        # sample
        # p_u = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":0, "l_m":0, "p_u":bias, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 1, 2, 4, 8, 9, 10])
    def measurement_bias_on_Y(bias):
        # measurement bias on Y (P_Y as target). Performance are calculated on Y
        # l_m_y = [0, 1, 2, 4, 8, 9, 10]
        param_dict = {"dim":dim, "l_y":0, "l_m_y":bias, "thr_supp":1, "l_h_r":0,  "l_h_q":0, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict, y_bias_meas=True)

    
    @bias_and_group_fairness_for_various_bias_values(bias_values=[1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0005])
    #@bias_and_group_fairness_for_various_bias_values(bias_values=[1, 0.5, 0.1])
    #@bias_and_group_fairness_for_various_bias_values(bias_values=[1, 0.5, 0.1, 0.05, 0.01])
    def representation_bias(bias):
        # representation bias
        # p_u = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":0, "l_m":0, "p_u":bias, "l_r":True, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    """
    # historical bias compensation
    # positive l_h_r and negative l_h_q
    param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":1.5,  "l_h_q":-1, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
    return pipeline(param_dict)
    """

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9])
    #@bias_and_group_fairness_for_various_bias_values(bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3])
    def historical_bias_on_R(bias):
        # historical bias on R
        # l_h_r = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":bias,  "l_h_q":0, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9])
    def historical_bias_on_Y(bias):
        # historical bias on Y
        # l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim":dim, "l_y":bias, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":0, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9])
    def historical_bias_on_Q(bias):
        # historical bias on Q
        # l_h_q = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim":dim, "l_y":0, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":bias, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9])
    def historical_bias_on_R_and_Y(bias):
        # historical bias on R and Y
        # l_h_r and l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim":dim, "l_y":bias, "l_m_y":0, "thr_supp":1, "l_h_r":bias,  "l_h_q":0, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9])
    def historical_bias_on_Q_and_Y(bias):
        # historical bias on Q and Y
        # l_h_q and l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim":dim, "l_y":bias, "l_m_y":0, "thr_supp":1, "l_h_r":0,  "l_h_q":bias, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)

    @bias_and_group_fairness_for_various_bias_values(bias_values=[0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9])
    def historical_bias_on_Q_and_R_and_Y(bias):
        # historical bias on Q and R and Y
        # l_h_r and l_h_q and l_y = [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
        param_dict = {"dim":dim, "l_y":bias, "l_m_y":0, "thr_supp":1, "l_h_r":bias,  "l_h_q":bias, "l_m":0, "p_u":1, "l_r":False, "l_o":False, "l_y_b":0, "l_q":2, "sy":2, "l_r_q":0}
        return pipeline(param_dict)
    


    no_bias()
    measurement_on_R()
    omission()
    undersampling()
    measurement_bias_on_Y()
    representation_bias()
    historical_bias_on_R()
    historical_bias_on_Y()
    historical_bias_on_Q()
    historical_bias_on_R_and_Y()
    historical_bias_on_Q_and_Y()
    historical_bias_on_Q_and_R_and_Y()


if __name__ == "__main__":
   # generate all plots
   bias_and_group_fairness()