![20180720101646!Intesa_Sanpaolo_logo](https://user-images.githubusercontent.com/92302358/187639073-08130658-5c1a-4f93-be2b-be180a30f38b.svg)

# ISParity
Repository of experiments in fairness Machine Learning.

> ### Authors & contributors:
> Alessandro Castelnovo, Riccardo Crupi, Daniele Regoli

To know more about this research work, please refer to our papers:

- [A clarification of the nuances in the fairness metrics landscape](https://www.nature.com/articles/s41598-022-07939-1)
  [ArXiv version](https://arxiv.org/pdf/2106.00467.pdf)
- Investigating Bias with a Synthetic Data Generator: Empirical Evidence and Philosophical Interpretation [work in progress]


![image](https://user-images.githubusercontent.com/66357086/202754476-9b270563-00b1-4f08-8404-de9396d67e0b.png)

### Using the biased dataset generator
Clone repo and install packages:
```
git clone https://github.com/rcrupiISP/ISParity.git
cd ISParity
pip install -r requirements.txt
```
Generate plots for different (magnitudes of) biases, fairness metrics, and bias mitigation techniques:
```
python plots.py
```

Python version: 3.8.10