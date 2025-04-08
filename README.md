# A Survey on Deep Text Hashing

This repository offers a carefully curated selection of research papers centered on **deep text hashing**. It is based on our survey paper, **A Survey on Deep Text Hashing: Efficient Semantic Text Retrieval with Binary Representation**. The list will be updated regularly. Should you come across any inaccuracies or overlooked works, you are warmly encouraged to open an issue or submit a pull request.

![](./image/framework.png)

## Category

- [Meaning of the Marker](#meaning_of_the_marker)
- [Paper List](#paper_list)
- [Datasets](#datasets_instruction)
- [Models](#models_instruction)
- [Acknowledgments](#acknowledgements)

## Meaning of the Marker

|  Marker    |  Meaning  |
| ---- | ---- |
|  ![](https://img.shields.io/badge/SemanticExtraction-Rec-brightgreen) |  Reconstruction-based method  |
|  ![](https://img.shields.io/badge/SemanticExtraction-Prior(X)-brightgreen) |  Applying A prior on the latent representation (X could be G: Gussain, B: Bernoulli, M: Mixture, C: Categorical, BM: Boltzmann, GA: Graph)|
|  ![](https://img.shields.io/badge/SemanticExtraction-Pse-brightgreen) |  Pseudo-similarity-based method |
|  ![](https://img.shields.io/badge/SemanticExtraction-MMI-brightgreen) |  Maximal mutual information method |
|  ![](https://img.shields.io/badge/SemanticExtraction-SFC-brightgreen) |  Learning semantic from categories |
|  ![](https://img.shields.io/badge/SemanticExtraction-SFR-brightgreen) |  Learning semantic from relevance |
|  ![](https://img.shields.io/badge/CodeQuality-CB-red) |  Promoting code balance |
|  ![](https://img.shields.io/badge/CodeQuality-FE-red) |  Promoting few-bit code |
|  ![](https://img.shields.io/badge/CodeQuality-Quan(X)-red) |  Using quantization method (X could be Loss: quantization loss, Sgn: Signum function, Sigmoid: Sigmoid function, Tanh: Tanh function, Stanh: The scaled tanh function) |
|  ![](https://img.shields.io/badge/OtherTechnology-Robustness-yellow) |  Promoting the robustness of hash codes |
|  ![](https://img.shields.io/badge/OtherTechnology-Gradient-yellow) |  Optimization of gradients during the backpropagation process in discrete layers. |
|  ![](https://img.shields.io/badge/OtherTechnology-Index-yellow) |  Adaptation to hashing index |
## Paper List

+ **De-confusing Hard Samples for Text Semantic Hashing.** In **ICASSP'2025**
[Paper](https://ieeexplore.ieee.org/abstract/document/10889846).\
![](https://img.shields.io/badge/SemanticExtraction-Rec,Prior(B),SFC,SFR-brightgreen)
![](https://img.shields.io/badge/CodeQuality-CB,Quan(Sgn)-red)
+ **Document Hashing with Multi-Grained Prototype-Induced Hierarchical Generative Model.** In **EMNLP'2024** [Paper](https://aclanthology.org/2024.findings-emnlp.18.pdf).\
![](https://img.shields.io/badge/SemanticExtraction-Rec,Prior(G),MMI,Pse-brightgreen)

## Datasets
Here, we have compiled a selection of widely utilized benchmark datasets for text hashing research. These datasets span diverse domains and exhibit a range of characteristics in terms of scale, label types, and download link. For a detailed introduction to the dataset, please refer to our survey.


|  Datasets | Instance | Categories | Single-/Multi-Label |    Link |
| ---- | ---- | ---- | ---- | ---- |
|   20Newsgroups   |   18,846   |   20   | single-label    |   [link](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)   |
|   Agnews   |   127,600   |  4  |  single-label    |   [link](http://groups.di.unipi.it/gulli/AG_corpus_of_news_articles.html)   |
|   Reuters   |  10,788    |   90/20   |   muti-label   |   [link](https://www.nltk.org/book/ch02.html)   |
|   DBpedia   |  60,000    |   14   |  single-label    |   [link](https://www.csie.ntu.edu.tw/cjlin/libsvmtools/datasets/multilabel.html)   |
|   RCV1   |   804,414   |  103/4   |   muti-label   |   [link](https://catalog.data.gov/dataset/siam-2007-text-mining-competition-dataset)   |
|   TMC   |   28,596   |   22   |  muti-label    |  [link](https://catalog.data.gov/dataset/siam-2007-text-mining-competition-dataset)    |
|   NYT   |   11,527   |  26    |  single-label    |   [link](https://emilhvitfeldt.github.io/textdata/reference/dataset_dbpedia.html)   |
|   Yahooanswer   |  1,460,000    |  10    |   single-label   |   [link](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset)   |


## Models

We have implemented several deep text hashing models using the PyTorch framework, while the implementation or migration of some other models is still underway. Our foundational code structure is inspired by the [VDSH](https://github.com/bayesquant/VDSH) repository. You can effortlessly run these codes.

**Note:** *Due to variations in data preprocessing, the results of different models may deviate from those reported in the original papers. We are actively working to standardize both the data processing pipeline and evaluation metrics to ensure a fairer and more consistent comparison.*

You can easily install the environment by

```bash
pip install . -r requirements.txt
```

Then, refer to the code in the *utils* folder to preprocess the dataset. Once the data preparation is complete, you can easily train and test any algorithm just by

```bash
sh models/{model_name}/run.sh
```

## Acknowledgments