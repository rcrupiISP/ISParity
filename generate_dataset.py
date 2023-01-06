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
        Lambda coefficient for measurement bias. If l_m!=0 P substitute R.
    p_u: float, optional
        Percentage of undersampling instance with A=1
    l_r: bool, optional
        Boolean for inducing representation bias, that is undersampling conditioning on a variable, e.g. X2
    l_o: bool, optional
        Boolean variable for excluding an important variable, e.g. X2
    l_y_b: float, optional
        Lambda coefficient for interaction proxy bias
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

    # Udersample
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
