![20180720101646!Intesa_Sanpaolo_logo](https://user-images.githubusercontent.com/92302358/187639073-08130658-5c1a-4f93-be2b-be180a30f38b.svg)

# ISParity
Repository of experiments in fairness Machine Learning.

> ### Authors & contributors:
> Joachim Baumann, Alessandro Castelnovo, Riccardo Crupi, Daniele Regoli

To know more about this research work, please refer to our papers:

- [A clarification of the nuances in the fairness metrics landscape](https://www.nature.com/articles/s41598-022-07939-1)
  [ArXiv version](https://arxiv.org/pdf/2106.00467.pdf)
- Bias on Demand: A Modelling Framework That Generates Synthetic Data With Bias


![image](https://user-images.githubusercontent.com/66357086/202754476-9b270563-00b1-4f08-8404-de9396d67e0b.png)

### Using the biased dataset generator
Clone repo and install packages:
```
git clone https://github.com/rcrupiISP/ISParity.git
cd ISParity
pip install -r requirements.txt
```

Generate a biased dataset that is saved as `my_biased_dataset.csv` with:
```
python generate_dataset.py -f my_biased_dataset
```
The following command line arguments are available to specify the types of biases to be introduced in the dataset:
- **dim**: **? bias**: ???
- **l_y**: **? bias**: ???
- **l_m_y**: **? bias**: ???
- **thr_supp**: **? bias**: ???
- **l_h_r**: **? bias**: ???
- **l_h_q**: **? bias**: ???
- **l_m**: **? bias**: ???
- **p_u**: **? bias**: ???
- **l_r**: **? bias**: ???
- **l_o**: **? bias**: ???
- **l_y_b**: **? bias**: ???
- **l_q**: **? bias**: ???
- **sy**: **? bias**: ???
- **l_r_q**: **? bias**: ???

Generate plots for different (magnitudes of) biases, fairness metrics, and bias mitigation techniques with:
```
python plots.py
```

Python version: 3.8.10