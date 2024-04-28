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

  - [An Open-Source Toolkit to Generate Biased Datasets](https://ceur-ws.org/Vol-3442/paper-02.pdf)  
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
  

- [Fundamental Rights and Artificial Intelligence Impact Assessment: A New Quantitative Methodology in the Upcoming Era of Ai Act](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4698609)
Bertaina, Samuele and Biganzoli, Ilaria and Desiante, Rachele and Fontanella, Dario and Inverardi, Nicole and Penco, Ilaria Giuseppina and Cosentini, Andrea. 
Fundamental Rights and Artificial Intelligence Impact Assessment: A New Quantitative Methodology in the Upcoming Era of Ai Act. 
 	> The upcoming Artificial Intelligence Act should require that operators of Artificial Intelligence (AI) systems perform a Fundamental Rights Impact Assessment (FRIA) when such systems are classified as high-risk as defined in Annex III of the proposed regulation. The aim of this work is to offer a comprehensive framework, specifically thought for private businesses, to assess the impact of AI systems on Fundamental Rights (FR) – also named human rights – of individuals. In a nutshell, the assessment approach that we propose consists of two stages: (1) an open-ended survey that helps gather the contextual information and the technical features, in order to properly identify potential threats for FR, and (2) a quantitative matrix that considers each right guaranteed by the European Charter of Fundamentals Rights and tries to measure the potential impacts with a traceable and robust procedure. In light of an increasingly pervasive use of AI systems and considering the specificity of such technologies, we believe that a structured and quantitative process for assessing the impact on FR of individuals is still lacking and could be of great importance in discovering and remedying possible violations. Indeed, the proposed framework could allow to: (1) be accountable and transparent in assessing the risks of implementing AI systems that affect people; (2) gain insights to understand if any right is threatened or any group of people is more vulnerable; (3) put in place, if necessary, remediation strategies before the deployment of AI systems through demonstrable mitigative actions, with the aim of being compliant with the regulation and limiting reputational damage.

- [Preserving Utility in Fair Top-k Ranking with Intersectional Bias](https://link.springer.com/chapter/10.1007/978-3-031-37249-0_5)
Nicola Alimonda, Alessandro Castelnovo, Riccardo Crupi, Fabio Mercorio, Mario Mezzanzanica
International Workshop on Algorithmic Bias in Search and Recommendation, 59-73
  	> Ranking is required for many real applications, such as search, personalisation, recommendation, and filtering. Recent research has focused on developing reliable ranking algorithms that maintain fairness in their outcomes. However, only a few consider multiple protected groups since this extension introduces significant challenges. While useful in the research sector, considering only one binary sensitive feature for handling fairness is inappropriate when the algorithm must be deployed responsibly in real-world applications. Our work is built on top of Multinomial FA*IR, a Fair Top-k ranking with multiple protected groups, which we extend to provide users the option to balance fairness and utility, adapting the final ranking accordingly. Our experimental results show that alternative better solutions overlooked by Multinomial FA*IR may be found through our approach without violating fairness boundaries. The code of the implemented solution and the experiments are publicly available to the community as a GitHub repository.

- [Marrying LLMs with Domain Expert Validation for Causal Graph Generation](https://ceur-ws.org/Vol-3650/paper7.pdf)
Alessandro Castelnovo, Riccardo Crupi, Fabio Mercorio, Mario Mezzanzanica, Daniele Potertì, Daniele Regoli		 
3rd Italian Workshop on Artificial Intelligence and Applications for Business and Industries - AIABI
co-located with AI*IA 2023
	> In the era of rapid growth and transformation driven by artificial intelligence across various sectors, which is catalyzing the fourth industrial revolution, this research is directed toward harnessing its potential to enhance the efficiency of decision-making processes within organizations. When constructing machine learning-based decision models, a fundamental step involves the conversion of domain knowledge into causal-effect relationships that are represented in causal graphs. This process is also notably advantageous for constructing explanation models. We present a method for generating causal graphs that integrates the strengths of Large Language Models (LLMs) with traditional causal theory algorithms. Our method seeks to bridge the gap between AI’s theoretical potential and practical applications. In contrast to recent related works that seek to exclude the involvement of domain experts, our method places them at the forefront of the process. We present a novel pipeline that streamlines and enhances domain-expert validation by providing robust causal graph proposals. These proposals are enriched with transparent reports that blend foundational causal theory reasoning with explanations from LLMs.

- [Fair Enough? A map of the current limitations of the requirements to have" fair''algorithms](https://arxiv.org/abs/2311.12435)
Alessandro Castelnovo, Nicole Inverardi, Gabriele Nanino, Ilaria Giuseppina Penco, Daniele Regoli
  	> In the recent years, the raise in the usage and efficiency of Artificial Intelligence and, more in general, of Automated Decision-Making systems has brought with it an increasing and welcome awareness of the risks associated with such systems. One of such risks is that of perpetuating or even amplifying bias and unjust disparities present in the data from which many of these systems learn to adjust and optimise their decisions. This awareness has on one side encouraged several scientific communities to come up with more and more appropriate ways and methods to assess, quantify, and possibly mitigate such biases and disparities. On the other hand, it has prompted more and more layers of society, including policy makers, to call for ``fair'' algorithms. We believe that while a lot of excellent and multidisciplinary research is currently being conducted, what is still fundamentally missing is the awareness that having ``fair'' algorithms is per s\'e a nearly meaningless requirement, that needs to be complemented with a lot of additional societal choices to become actionable. Namely, there is a hiatus between what the society is demanding from Automated Decision-Making systems, and what this demand actually means in real-world scenarios. In this work, we outline the key features of such a hiatus, and pinpoint a list of fundamental ambiguities and attention points that we as a society must address in order to give a concrete meaning to the increasing demand of fairness in Automated Decision-Making systems.


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

- [DTOR: Decision Tree Outlier Regressor to explain anomalies](https://arxiv.org/abs/2403.10903) 
 Riccardo Crupi, Daniele Regoli, Alessandro Damiano Sabatino, Immacolata Marano, Massimiliano Brinis, Luca Albertazzi, Andrea Cirillo, Andrea Claudio Cosentini
[GitHub](https://github.com/rcrupiISP/DTOR)
	> Explaining outliers occurrence and mechanism of their occurrence can be extremely important in a variety of domains. Malfunctions, frauds, threats, in addition to being correctly identified, oftentimes need a valid explanation in order to effectively perform actionable counteracts. The ever more widespread use of sophisticated Machine Learning approach to identify anomalies make such explanations more challenging. We present the Decision Tree Outlier Regressor (DTOR), a technique for producing rule-based explanations for individual data points by estimating anomaly scores generated by an anomaly detection model. This is accomplished by first applying a Decision Tree Regressor, which computes the estimation score, and then extracting the relative path associated with the data point score. Our results demonstrate the robustness of DTOR even in datasets with a large number of features. Additionally, in contrast to other rule-based approaches, the generated rules are consistently satisfied by the points to be explained. Furthermore, our evaluation metrics indicate comparable performance to Anchors in outlier explanation tasks, with reduced execution time.

- [Quantifying credit portfolio sensitivity to asset correlations with interpretable generative neural networks](https://doi.org/10.21314/JRMV.2024.002)
  Sergio Caprioli, Emanuele Cagliero and Riccardo Crupi
Journal of Risk Model Validation - ISSN: 1753-9579 (print) 1753-9587 (online)
[GitHub](https://github.com/rcrupiISP/SyntheticCorrelationVAE)
	> We propose a novel approach for quantifying the sensitivity of credit portfolio value-at-risk to asset correlations with the use of synthetic financial correlation matrixes generated with deep learning models. In previous work, generative adversarial networks (GANs) were employed to demonstrate the generation of plausible correlation matrixes that capture the essential characteristics observed in empirical correlation matrixes estimated on asset returns. Instead of GANs, we employ variational autoencoders (VAEs) to achieve a more interpretable latent space representation and to obtain a generator of plausible correlation matrixes by sampling the VAE’s latent space. Through our analysis, we reveal that the VAE’s latent space can be a useful tool to capture the crucial factors impacting portfolio diversification, particularly in relation to the sensitivity of credit portfolios to changes in asset correlations. A VAE trained on the historical time series of correlation matrixes is used to generate synthetic correlation matrixes that satisfy a set of expected financial properties. Our analysis provides clear indications that the capacity for realistic data augmentation provided by VAEs, combined with the ability to obtain model interpretability, can prove useful for risk management, enhancing the resilience and accuracy of models when backtesting, as past data may exhibit biases and might not contain the essential high-stress events required for evaluating diverse risk scenarios.

### Other topics

- [Disambiguation of company names via deep recurrent networks](https://www.sciencedirect.com/science/article/abs/pii/S095741742302537X)
Alessandro Basile, Riccardo Crupi, Michele Grasso, Alessandro Mercanti, Daniele Regoli, Simone Scarsi, Shuyi Yang, Andrea Claudio Cosentini.	
Expert Systems with Applications 238, 122035.
[GitHub](https://github.com/rcrupiISP/SiameseDisambiguation)
	> Name Entity Disambiguation is the Natural Language Processing task of identifying textual records corresponding to the same Named Entity, i.e., real-world entities represented as a list of attributes (names, places, organisations, etc.). In this work, we face the task of disambiguating companies on the basis of their written names. We propose a Siamese LSTM Network approach to extract – via supervised learning – an embedding of company name strings in a (relatively) low dimensional vector space and use this representation to identify pairs of company names that actually represent the same company (i.e., the same Entity).
Given that the manual labelling of string pairs is a rather onerous task, we analyse how an Active Learning approach to prioritise the samples to be labelled leads to a more efficient overall learning pipeline.
The contributions of this work are: with empirical investigations on real-world industrial data, we show that our proposed Siamese Network outperforms several benchmark approaches based on standard string matching algorithms when enough labelled data are available; moreover, we show that Active Learning prioritisation is indeed helpful when labelling resources are limited, and let the learning models reach the out-of-sample performance saturation with less labelled data with respect to standard (random) data labelling approaches.


## Lectures, Tutorials, and others

[Tutorial Fairness and Explainability Machine learning Milano - Alessandro Castelnovo 2020](https://youtu.be/1WLv09HE2j8)

[CEILS talk Causal UAI - Riccardo Crupi 2021](https://youtu.be/adTNX_Um47I)

[Seminar BeFair - Riccardo Crupi 2022](https://uniudamce-my.sharepoint.com/:v:/g/personal/155794_spes_uniud_it/ESo3k-cagu9Pqfm8VVSgragBxQpxUnvDrrwtJ8_ZhgVAUg?e=SWlAR9)

[Fairness in AI Deep Learning Italia - Daniele Regoli 2022](https://youtu.be/yOzJzBYab7I)

[Tutorial FFTree - 2022](https://sites.google.com/campus.unimib.it/fftree/home)

[Toward Fairness Through Time BIAS@ECMLPKDD - Alessandro Castelnovo 2022](https://youtu.be/hmCwg4lg8BY)

[Demo BeFair: Addressing Fairness in the Banking Sector - Greta Greco 2022](https://www.youtube.com/watch?v=nW6O444EbDQ)

[Preserving fairness in ranking BIAS@ECIR2023 - Nicola Alimonda 2023](https://youtu.be/l9QsCyYZgg8)

[BiasOnDemand ACM FAccT - Daniele Regoli 2023](https://youtu.be/6pCN8cOHEBc)
