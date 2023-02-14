import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_synth(dim=15000, l_y=4, l_m_y=0, thr_supp=1, l_h_r=1.5,  l_h_q=1, l_m=1, p_u=1, l_r=False, l_o=False, l_y_b=0, l_q=2, sy=5, l_r_q=0):
    """Generate a synthetic dataset.

    Parameters
    ----------
    dim : int
        Dimension of the dataset
    l_y : float, optional
        Lambda coefficient for historical bias on the target y
    l_m_y : float, optional
        Lambda coefficient for measurement bias on the target y
    thr_supp: float, optional
        Threshold correlation for discarding features too much correlated with s
    l_h_r: float, optional
        Lambda coefficient for historical bias on R
    l_h_q: float, optional
        Lambda coefficient for historical bias on Q
    l_m: float, optional
        Lambda coefficient for measurement bias. If l_m!=0 P substitutes R.
    p_u: float, optional
        Percentage of undersampling instance with A=1
    l_r: bool, optional
        Boolean for inducing representation bias, that is undersampling conditioning on a variable, e.g. R
    l_o: bool, optional
        Boolean variable for excluding an important variable, e.g. X2
    l_y_b: float, optional
        Lambda coefficient for interaction proxy bias, i.e., historical bias on the label y with lower values of y for individuals in group A=1 with high values for the feature R
    l_q: float, optional
        Lambda coefficient for importance of Q for Y
    sy: float, optional
        Standard deviation of the noise of Y
    l_r_q: float, optional
        Lambda coefficient that quantifies the influence from R to Q

    Returns
    -------
    list
        a list datasets train and test for: complete dataset, individual and suppression.
    """
    np.random.seed(42)
    # # # dataset
    # Random var
    N1 = np.random.gamma(2, 3, dim)
    N2 = np.random.normal(1, 0.5, dim)
    Np = np.random.normal(0, 2, dim)
    Ny = np.random.normal(0, sy, dim)
    A = np.random.binomial(1, 0.5, size=dim)

    # X var
    # Variable R defined as the salary
    R = N1 - l_h_r*A
    # Variable R influence Q, a discrete variable that define a zone in a city
    R_A = 1/(1 + np.exp(l_r_q*R - l_h_q*A))
    Q = np.random.binomial(3, R_A)

    # Y var
    # y target, with measurement and historical bias
    y = R - l_q*Q - l_y*A - l_m_y*A + Ny + l_y_b*R*A
    # y only historical, no measurement bias
    y_real = R - l_q*Q - l_y*A + Ny + l_y_b*R*A

    if l_m != 0:
        # Proxy for R, e.g. the dimension of the house
        P = R - A*l_m + Np
        print("Correlation between R and P: ", np.corrcoef(P, R))
        dtf = pd.DataFrame({'P': P, 'Q': Q, 'A': A, 'Y': y, 'Y_real': y_real})
    else:
        dtf = pd.DataFrame({'R': R, 'Q': Q, 'A': A, 'Y': y, 'Y_real': y_real})

    # Undersample
    int_p_u = int(((dtf['A'] == 1).sum())*p_u)
    if int_p_u > 0:
        if l_r:
            # Undersample with increasing R, for A=1 the people will results poor
            drop_index = dtf.loc[dtf['A'] == 1, :].sort_values(
                by='R', ascending=True).index
            dtf = dtf.drop(drop_index[int_p_u:])
        else:
            dtf = dtf.drop(dtf.index[dtf['A'] == 1][int_p_u:])
    # Delete an important variable for omission: R or P
    if l_o:
        if 'R' in dtf.columns:
            del dtf['R']
        elif 'P' in dtf.columns:
            del dtf['P']
        else:
            print("Condition non possible. How I could get here?")
    # Define feature matrix X and target Y
    X = dtf.reset_index(drop=True)
    y = X['Y']
    y_real = X['Y_real']
    del X['Y']
    del X['Y_real']
    # Define threshold making y binary
    thres = y.mean()
    y = pd.Series(1*(y > thres))
    y_real = pd.Series(1*(y_real > thres))
    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=X['A'] == 1)
    # individual set
    X_ind_train = X_train[[i for i in X_train.columns if i != 'A']]
    X_ind_test = X_test[[i for i in X_test.columns if i != 'A']]
    # suppression set
    X_supp_train = X_train[[i for i in X_train.columns if i != 'A' and
                            abs(np.corrcoef(X_train[i], X_train['A'])[0, 1]) < thr_supp]]
    X_supp_test = X_test[[i for i in X_test.columns if i != 'A' and
                          abs(np.corrcoef(X_train[i], X_train['A'])[0, 1]) < thr_supp]]
    # get y_test not biased
    y_train_real = y_real[y_train.index]
    y_test_real = y_real[y_test.index]

    return X_train, X_ind_train, X_supp_train, X_test, X_ind_test, X_supp_test, y_train, y_test, y_train_real, y_test_real


def main():

    # generate unbiased dataset: python generate_dataset.py -f my_unbiased_dataset
    # generate biased dataset: python generate_dataset.py -f my_biased_dataset TODO
    # get help with: python generate_dataset.py -h

    parser = argparse.ArgumentParser(
        description='Generating a biased dataset.', argument_default=argparse.SUPPRESS)

    parser.add_argument('-p', '--path', type=str, required=False,
                        help='The name of the directory where the new dataset should be stored. Existing csv files are overwritten')

    # dataset properties
    parser.add_argument('-dim', type=int, required=False,
                        help='Dimension of the dataset')
    parser.add_argument('-sy', type=float, required=False,
                        help='Standard deviation of the noise of Y')
    parser.add_argument('-l_q', type=float, required=False,
                        help='Lambda coefficient for importance of Q for Y')
    parser.add_argument('-l_r_q', type=float, required=False,
                        help='Lambda coefficient that quantifies the influence from R to Q')
    parser.add_argument('-thr_supp', type=float, required=False,
                        help='Threshold correlation for discarding features too much correlated with s')
    # biases
    parser.add_argument('-l_y', type=float, required=False,
                        help='Lambda coefficient for historical bias on the target y')
    parser.add_argument('-l_m_y', type=float, required=False,
                        help='Lambda coefficient for measurement bias on the target y')
    parser.add_argument('-l_h_r', type=float, required=False,
                        help='Lambda coefficient for historical bias on R')
    parser.add_argument('-l_h_q', type=float, required=False,
                        help='Lambda coefficient for historical bias on Q')
    parser.add_argument('-l_m', type=float, required=False,
                        help='Lambda coefficient for measurement bias on the feature R. If l_m!=0 P substitutes R.')
    parser.add_argument('-p_u', type=float, required=False,
                        help='Percentage of undersampling instance with A=1')
    parser.add_argument('-l_r', type=str, required=False,
                        help='Boolean for inducing representation bias, that is undersampling conditioning on a variable, e.g. R')
    parser.add_argument('-l_o', type=str, required=False,
                        help='Boolean variable for excluding an important variable (ommited variable bias), e.g. R (or its proxy)')
    parser.add_argument('-l_y_b', type=float, required=False,
                        help='Lambda coefficient for interaction proxy bias, i.e., historical bias on the label y with lower values of y for individuals in group A=1 with high values for the feature R')

    args = parser.parse_args()
    args = vars(args)

    path = args.pop('path', None)
    if path is None:
        path = 'my_new_dataset'
    path = f'datasets/{path}'
    # create directory if it does not exist yet
    Path(path).mkdir(parents=True, exist_ok=True)

    if 'l_r' in args.keys():
        args['l_r'] = args['l_r'].lower() in ['true', '1', 1]
    if 'l_o' in args.keys():
        args['l_o'] = args['l_o'].lower() in ['true', '1', 1]

    # generat the (biased) dataset
    X_train, X_ind_train, X_supp_train, X_test, X_ind_test, X_supp_test, y_train, y_test, y_train_real, y_test_real = create_synth(
        **args)

    # save the generated dataset
    X_train.to_csv(Path(path).joinpath('X_train.csv'))
    X_ind_train.to_csv(Path(path).joinpath('X_ind_train.csv'))
    X_supp_train.to_csv(Path(path).joinpath('X_supp_train.csv'))
    X_test.to_csv(Path(path).joinpath('X_test.csv'))
    X_ind_test.to_csv(Path(path).joinpath('X_ind_test.csv'))
    X_supp_test.to_csv(Path(path).joinpath('X_supp_test.csv'))
    y_train.to_csv(Path(path).joinpath('y_train.csv'))
    y_test.to_csv(Path(path).joinpath('y_test.csv'))
    y_train_real.to_csv(Path(path).joinpath('y_train_real.csv'))
    y_test_real.to_csv(Path(path).joinpath('y_test_real.csv'))

    print(
        f'\n:)\n:) The dataset has been generated and saved in the directory {path}/\n:)')


if __name__ == "__main__":
    main()
