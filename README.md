![20180720101646!Intesa_Sanpaolo_logo](https://user-images.githubusercontent.com/92302358/187639073-08130658-5c1a-4f93-be2b-be180a30f38b.svg)

# ISParity
Repository of experiments in fairness Machine Learning.  
:star: **Data Science & Artificial Intelligence** :star: group @ Intesa Sanpaolo

---

> ### Authors & contributors:
> Alessandro Castelnovo, Riccardo Crupi, Daniele Regoli

![image](https://user-images.githubusercontent.com/66357086/202754476-9b270563-00b1-4f08-8404-de9396d67e0b.png)

---

To know more about the research work of our team of :star: **Data Science & Artificial Intelligence** :star: in Intesa Sanpaolo, please refer to our publications
## Journal papers & Conference proceedings 

### Fairness

- [Bias on Demand: A Modelling Framework That Generates Synthetic Data With Bias](https://doi.org/10.1145/3593013.3594058)   
Joachim Baumann, Alessandro Castelnovo, Riccardo Crupi, Nicole Inverardi, Daniele Regoli    
FAccT '23: Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency June 2023, Pages 1002–1013 [(video presentation at FAccT)](https://www.youtube.com/watch?v=6pCN8cOHEBc&feature=youtu.be)
	> Nowadays, Machine Learning (ML) systems are widely used in various businesses and are increasingly being adopted to make decisions that can significantly impact people’s lives. However, these decision-making systems rely on data-driven learning, which poses a risk of propagating the bias embedded in the data. Despite various attempts by the algorithmic fairness community to outline different types of bias in data and algorithms, there is still a limited understanding of how these biases relate to the fairness of ML-based decision-making systems. In addition, efforts to mitigate bias and unfairness are often agnostic to the specific type(s) of bias present in the data. This paper explores the nature of fundamental types of bias, discussing their relationship to moral and technical frameworks. To prevent harmful consequences, it is essential to comprehend how and where bias is introduced throughout the entire modelling pipeline and possibly how to mitigate it. Our primary contribution is a framework for generating synthetic datasets with different forms of biases. We use our proposed synthetic data generator to perform experiments on different scenarios to showcase the interconnection between biases and their effect on performance and fairness evaluations. Furthermore, we provide initial insights into mitigating specific types of bias through post-processing techniques. The implementation of the synthetic data generator and experiments can be found at https://github.com/rcrupiISP/BiasOnDemand.

* [An Open-Source Toolkit to Generate Biased Datasets](https://ceur-ws.org/Vol-3442/paper-02.pdf)  
Joachim Baumann, Alessandro Castelnovo, Riccardo Crupi, Nicole Inverardi, Daniele Regoli  
EWAF’23: European Workshop on Algorithmic Fairness, June 07–09, 2023, Winterthur, Switzerland
	> Many different types of bias are discussed in the algorithmic fairness community. A clear understanding of those biases and their relation to fairness metrics and mitigation techniques is still missing. We introduce Bias on Demand: a modelling framework to generate synthetic datasets that contain various types of bias. Furthermore, we clarify the effect of those biases on the accuracy and fairness of ML systems and provide insights into the trade-offs that emerge when trying to mitigate them. We believe that our open-source package will enable researchers and practitioners to better understand and mitigate different types of biases throughout the ML pipeline. The package can be installed via pip and the experiments are available at https://github.com/rcrupiISP/BiasOnDemand. We encourage readers to consult the full paper.

- [A clarification of the nuances in the fairness metrics landscape](https://doi.org/10.1038/s41598-022-07939-1)    
Alessandro Castelnovo, Riccardo Crupi, Greta Greco, Daniele Regoli, Ilaria Penco, Andrea Cosentini  
*Scientific Reports* **12**, 4209 (2022)
	> In recent years, the problem of addressing fairness in machine learning (ML) and automatic decision making has attracted a lot of attention in the scientific communities dealing with artificial intelligence. A plethora of different definitions of fairness in ML have been proposed, that consider different notions of what is a “fair decision” in situations impacting individuals in the population. The precise differences, implications and “orthogonality” between these notions have not yet been fully analyzed in the literature. In this work, we try to make some order out of this zoo of definitions.  

- [Towards Responsible AI: A Design Space Exploration of Human-Centered Artificial Intelligence User Interfaces to Investigate Fairness](https://doi.org/10.1080/10447318.2022.2067936)  
Yuri Nakao, Lorenzo Strappelli, Simone Stumpf, Aisha Naseer, Daniele Regoli, Giulia Del Gamba  
*International Journal of Human–Computer Interaction* 2022
	> With Artificial intelligence (AI) to aid or automate decision-making advancing rapidly, a particular concern is its fairness. In order to create reliable, safe and trustworthy systems through human-centred artificial intelligence (HCAI) design, recent efforts have produced user interfaces (UIs) for AI experts to investigate the fairness of AI models. In this work, we provide a design space exploration that supports not only data scientists but also domain experts to investigate AI fairness. Using loan applications as an example, we held a series of workshops with loan officers and data scientists to elicit their requirements. We instantiated these requirements into FairHIL, a UI to support human-in-the-loop fairness investigations, and describe how this UI could be generalized to other use cases. We evaluated FairHIL through a think-aloud user study. Our work contributes better designs to investigate an AI model’s fairness—and move closer towards responsible AI.


* [FFTree: A flexible tree to handle multiple fairness criteria](https://doi.org/10.1016/j.ipm.2022.103099)    
Alessandro Castelnovo, Andrea Cosentini, Lorenzo Malandri, Fabio Mercorio, Mario Mezzanzanica  
*Information Processing & Management* 2022  
	> The demand for transparency and fairness in AI-based decision-making systems is constantly growing. Organisations need to be assured that their applications, based on these technologies, behave fairly, without introducing negative social implications in relation to sensitive attributes such as gender or race. Since the notion of fairness is context dependent and not uniquely defined, studies in the literature have proposed various formalisation. In this work, we propose a novel, flexible, discrimination-aware decision-tree that allows the user to employ different fairness criteria depending on the application domain. Our approach enhances decision-tree classifiers to provide transparent and fair rules to final users.

- [BeFair: Addressing Fairness in the Banking Sector](https://doi.org/10.1109/BigData50022.2020.9377894)    
Alessandro Castelnovo, Riccardo Crupi, Giulia Del Gamba, Greta Greco, Aisha Naseer, Daniele Regoli, Beatriz San Miguel Gonzalez  
2020 IEEE International Conference on Big Data (Big Data)  
[(preprint version)](https://arxiv.org/pdf/2102.02137)
	> Algorithmic bias mitigation has been one of the most difficult conundrums for the data science community and Machine Learning (ML) experts. Over several years, there have appeared enormous efforts in the field of fairness in ML. Despite the progress toward identifying biases and designing fair algorithms, translating them into the industry remains a major challenge. In this paper, we present the initial results of an industrial open innovation project in the banking sector: we propose a general roadmap for fairness in ML and the implementation of a toolkit called BeFair that helps to identify and mitigate bias. Results show that training a model without explicit constraints may lead to bias exacerbation in the predictions.
  

### eXplainable AI
- [Counterfactual explanations as interventions in latent space](https://doi.org/10.1007/s10618-022-00889-2)   
Riccardo Crupi, Alessandro Castelnovo, Daniele Regoli, Beatriz San Miguel Gonzalez  
*Data Mining and Knowledge Discovery* (2022)  
[(prerpint version)](https://arxiv.org/abs/2106.07754)  
   > Explainable Artificial Intelligence (XAI) is a set of techniques that allows the understanding of both technical and non-technical aspects of Artificial Intelligence (AI) systems. XAI is crucial to help satisfying the increasingly important demand of _trustworthy_ Artificial Intelligence, characterized by fundamental aspects such as respect of human autonomy, prevention of harm, transparency, accountability, etc. Within XAI techniques, counterfactual explanations aim to provide to end users a set of features (and their corresponding values) that need to be changed in order to achieve a desired outcome. Current approaches rarely take into account the feasibility of actions needed to achieve the proposed explanations, and in particular, they fall short of considering the causal impact of such actions. In this paper, we present Counterfactual Explanations as Interventions in Latent Space (CEILS), a methodology to generate counterfactual explanations capturing by design the underlying causal relations from the data, and at the same time to provide feasible recommendations to reach the proposed profile. Moreover, our methodology has the advantage that it can be set on top of existing counterfactuals generator algorithms, thus minimising the complexity of imposing additional causal constrains. We demonstrate the effectiveness of our approach with a set of different experiments using synthetic and real datasets (including a proprietary dataset of the financial domain).

* [Leveraging Causal Relations to Provide Counterfactual Explanations and Feasible Recommendations to End Users](https://www.scitepress.org/Papers/2022/107615/107615.pdf)  
Riccardo Crupi, Alessandro San Miguel González, Beatriz: Castelnovo, Daniele Regoli   
Proceedings of the 14th International Conference on Agents and Artificial Intelligence (ICAART 2022) - Volume 2, pages 24-32  
    > Over the last years, there has been a growing debate on the ethical issues of Artificial Intelligence (AI). Explainable Artificial Intelligence (XAI) has appeared as a key element to enhance trust of AI systems from both technological and human-understandable perspectives. In this sense, counterfactual explanations are becoming a de facto solution for end users to assist them in acting to achieve a desired outcome. In this paper, we present a new method called Counterfactual Explanations as Interventions in Latent Space (CEILS) to generate explanations focused on the production of feasible user actions. The main features of CEILS are: it takes into account the underlying causal relations by design, and can be set on top of an arbitrary counterfactual explanation generator. We demonstrate how CEILS succeeds through its evaluation on a real dataset of the financial domain.

### Other topics


## Lectures, Tutorials, and others

[Seminar BeFair - Riccardo Crupi 2022](https://uniudamce-my.sharepoint.com/personal/155794_spes_uniud_it/_layouts/15/stream.aspx?id=%2Fpersonal%2F155794%5Fspes%5Funiud%5Fit%2FDocuments%2FRegistrazioni%2FBeFair%5FUniud%2Emp4&ga=1
)

[CEILS talk Causal UAI - Riccardo Crupi 2021](https://youtu.be/adTNX_Um47I)

[Toward Fairness Through Time BIAS@ECMLPKDD - Alessandro Castelnovo 2022](https://youtu.be/hmCwg4lg8BY)


