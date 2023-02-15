![20180720101646!Intesa_Sanpaolo_logo](https://user-images.githubusercontent.com/92302358/187639073-08130658-5c1a-4f93-be2b-be180a30f38b.svg)

# ISParity
Repository of experiments in fairness Machine Learning.

> ### Authors & contributors:
> Joachim Baumann, Alessandro Castelnovo, Riccardo Crupi, Daniele Regoli

To know more about this research work, please refer to our papers:

- [A clarification of the nuances in the fairness metrics landscape](https://www.nature.com/articles/s41598-022-07939-1)
  [ArXiv version](https://arxiv.org/pdf/2106.00467.pdf)
- Bias on Demand: A Modelling Framework That Generates Synthetic Data With Bias


### Using the biased dataset generator
Clone repo and install packages:
```
git clone https://github.com/rcrupiISP/ISParity.git
cd ISParity
pip install -r requirements.txt
```

Python version: `3.8.10`

### Generating synthetic datasets (with bias)

Generate a dataset consisting of 1000 rows in total that is free of any bias and save it in the directory `datasets/my_unbiased_dataset/` with the following command:
```
python generate_dataset.py -p my_unbiased_dataset -dim 1000
```

The following command line arguments are available to specify properties of the dataset:
- **dim**: Dimension of the dataset
- **sy**: Standard deviation of the noise of Y
- **l_q**: Lambda coefficient for importance of Q for Y
- **l_r_q**: Lambda coefficient that quantifies the influence from R to Q
- **thr_supp**: Threshold correlation for discarding features too much correlated with s

Furthermore, the following command line arguments are available to specify the types of biases to be introduced in the dataset:
- **l_y**: Lambda coefficient for historical bias on the target y
- **l_m_y**: Lambda coefficient for measurement bias on the target y
- **l_h_r**: Lambda coefficient for historical bias on R
- **l_h_q**: Lambda coefficient for historical bias on Q
- **l_m**: Lambda coefficient for measurement bias on the feature R. If l_m!=0 P substitutes R.
- **p_u**: Percentage of undersampling instance with A=1
- **l_r**: Boolean for inducing representation bias, that is undersampling conditioning on a variable, e.g. R
- **l_o**: Boolean variable for excluding an important variable (ommited variable bias), e.g. R (or its proxy)
- **l_y_b**: Lambda coefficient for interaction proxy bias, i.e., historical bias on the label y with lower values of y for individuals in group A=1 with high values for the feature R

The biases are introduced w.r.t. idividuals in the group A=1.
For most types of bias, larger values mean more bias. The only exceptions are undersampling and representation bias (which can be seen as a specific type of undersampling conditional on the feature R) where smaller values correspond to more (conditional) undersampling, i.e., more bias.

For example, you can generate a dataset that includes measurement bias on the label Y and historical bias on the feature R with the following command:
```
python generate_dataset.py -p my_unbiased_dataset -dim 1000 
```
This dataset will be saved in `datasets/my_biased_dataset/`.

### Investigating bias, fairness, and mitigation techniques

Generate plots for different (magnitudes of) biases, fairness metrics, and bias mitigation techniques with:
```
python plots.py
```
This generates various plots for many different scenarios (i.e., datasets containing different magnitudes and types of bias). This might take a while. You can comment out the different types of plots to be generated at the end of the `plots.py` file. Alternativels, adjust the `bias_plots` function to only include preferred scenarios to speed it up.
