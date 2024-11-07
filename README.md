## Deep Text Hashing
The Python implementation of Some Deep Semantic Hashing Models. 

#### Getting Started

You can easily install the environment by

```bash
pip install . -r requirements.txt
```

You can easily train and test any algorithm just by

```bash
python VDSH.py
```

#### Datasets

We use the following five datasets for test:

**Agnews**: The Agnews dataset is a commonly used text classification dataset that contains news articles from Agence France-Presse (AFP). This dataset covers four main topics: World, National, Business, and Technology. It is a single-label dataset, meaning each news article is assigned only one topic label.

**Dbpedia**: DBpedia is a knowledge graph constructed based on Wikipedia, containing rich structured information such as entities, properties, and relationships. The DBpedia dataset is a collection of 630,000 documents classified into 14 non-overlapping ontology classes. It is typically a multi-label dataset, as each entity can have multiple attributes and relationships. The data is usually split into 560,000 training documents that serve as the database and 70,000 testing documents for querying. 
 
**20Newspaper**: The 20Newsgroups dataset is a classic text classification dataset consisting of newsgroup articles from 20 different topics. These topics cover various domains such as computer technology, sports, religion, among others. Each article is assigned to a specific newsgroup topic. The dataset comprises around 20,000 articles, and after preprocessing and cleaning, typically about 18,000 articles are used for training and testing. It is a single-label dataset, meaning each article has only one label indicating the newsgroup topic it belongs to. Generally, the 20Newsgroups dataset is randomly split into training, validation, and test sets. Typically, around 60\%-80\% of the data is used for training, 10%-20% for validation, and the remaining data for testing.

**Yahooanswer**: Yahoo Answers is a knowledge question-and-answer platform where users can ask questions and receive answers from other users. The platform covers a wide range of topics, including health, education, technology, entertainment, and more. It is usually a single-label dataset, meaning each question or answer is associated with only one label or category. The dataset typically includes questions collected from the Yahoo Answers platform along with their corresponding answers. After processing and cleaning, the dataset may contain 207,261 documents. 

