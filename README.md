# A Keyword-Extraction-Algorithm Based Searching Method for Subscription Articles
Cheng Zheng, Jinhua Sun, Yuliang Zhang, Yejia Liu 

2023/1/11


---------------------------

## Table of contents
--------------------

<!--ts-->
- [Table of Contents](#Table-of-Contents)
- [Background](#Background)
  - [Intro to 问需金山](#Intro-to-问需金山)
  - [Problem](#Problem)
  - [Motivation](#Motivation)
- [Core Concept Explanation](#Core-Concept-Explanation)
  - [Supervised Algorithm](#Supervised-Algorithm)
  - [TF-IDF](#TF-IDF)
  - [TextRank](#TextRank)
  - [Synonym](#Synonym) 
- [Algorithm Implementation](#Algorithm-Implementation)
  - [Overview](#Overview)
  - [Results](#Results)
    - [Keyword Searching](#Keyword-Searching)
    - [Full-Text Content Searching](#Full-Text-Content-Searching)
    - [Problem Solved](#Problem-Solved)
- [Analysis](#Analysis)
  -[Keyword Extraction Precision](#Keyword-Extraction-Precision)
  -[Efficiency for Keyword Searching](#Efficiency-for-Keyword-Searching)
- [Reference](#Reference)
- [Related Materials](#Related-Materials)
<!--te-->

## Background

### Intro to 问需金山

"**问需金山**" is a subsciption serving residents of **Jinshan community** in Fuzhou. It has the following functions:

+ It can help people learn about epidemic prevention policies, nucleic acid sites and other important information by pushing articles:

+ Providing benefits such as free fruit and medicine
+ A **你呼我应** platform to understand people's situation and solve people's difficulties which provides people with convenience.

### Problem

However, after our team actually used the official account, we found that there are two functions worth improving inside the official account.

+ First, the information articles in the 便民服务 module lack search function, and can only be sorted by time, as shown in the figure. It is not easy for people to look up important articles in the past.

<div align=center><img src="./these.images/image-20230111171901807.png" alt="image-20230111171901807" style="zoom:20%;" /></div>  

+ At the same time, the problem may become more prominent in the future as the number of articles increases.
+ Secondly, the function of follow automatic reply provided by the chat interface of the official account is not perfect. 
	+ At present, users can only get corresponding tweets by inputting the two keywords of community and nucleic acid

<div align=center><img src="./these.images/痛点2.1.JPEG" alt="痛点2.1" style="zoom:20%;" /></div>  

It was found that many of the keywords people entered in the chat box did not get the desired tweets:

| Rank | Keyword | Input Times |
| ---- | ------- | ----------- |
| 1    | 秒杀    | 608         |
| 2    | 核酸    | 381         |
| 3    | 社区    | 194         |
| 4    | 看病    | 39          |
| 5    | 金山    | 34          |
| 6    | 做      | 32          |
| 7    | 信息    | 31          |
| 8    | 健康    | 39          |
| 9    | 你好    | 26          |
| 10   | 检测    | 26          |

which also shows that the user support for this feature is actually very **high**.

### Motivation

Based on this, our team determine our project objectives. 

+ On the one hand, it is to realize the search function of information articles in the "convenience service".

+ On the other hand, it is to realize the expansion of the keyword database of "Pay attention to automatic reply".

Both of which require us to achieve **automatic keyword extraction**, which is the core of our whole project.

## Core Concept Explanation

This part will be an explanation for all the concepts used in this project.

### Supervised Algorithm

Keyphrase Extraction (KPE, or Keyword Extraction) can automatically extract phrases from an article and can summarize the core content in the document, which is beneficial to downstream information retrieval and other tasks. Keyword extraction uses machine learning, artificial intelligence (AI) with natural language processing (NLP) and there are mainly two methods for keyword extraction, **supervised** and **unsupervised** learning methods.

<div align=center><img src="./these.images/image-20230111152432933.png" alt="image-20230111152432933" style="zoom:30%;" /></div>  

For **unsupervised learning** like TF-IDF, TextRank and LDA, they mainly focus on statistical information of the article, while for **supervised learning** such as SVM, K-means, and Naive Bayes, labeling and supervising play a more important role.

At present, **unsupervised** keyword extraction is more widely used in practical applications due to less consumption of resources and manpower. However, due to the difference between the length of candicate words and document sequences, the representation of key phrase candidates and documents does not match, resulting in poor performance in long documents.

In our project, we use [**TextRank**](https://www.researchgate.net/publication/200042361_TextRank_Bringing_Order_into_Text) as simple keyword extraction for subscription articles, which can exhibit a relatively good performance comparing to human's work.

#### TF-IDF

The first keyword extraction algorithm is [**TF-IDF**](https://www.researchgate.net/publication/228818851_Using_TF-IDF_to_determine_word_relevance_in_document_queries). Through **Term Frequency** and **Inverse Document Frequency**, we can infer which words may be the key words of the article. The general equation for TF-IDF is  
<div align=center><img src="./these.images/公式1.png" alt="公式1" style="zoom:100%;" /></div>  
while d means a single document and D means document corpus. For the weight w of each word, it not only considers the word frequency in the **current** document, but also considers the word frequency in the **whole** corpus. Therefore, words such as **articles** and **pronouns** can be screened out from the list of key words

In python coding, we import `jieba` package and implement the algorithm easily. In order to see the result of the algorithm, we use [this](https://mp.weixin.qq.com/s/JqxjRVQxflbb8J-oXM5nFw) article as a testcase to demonstrate the idea of the algorithm, and below is our test results. Notice that there is a *weight* colume on the right, which indicates the **probability** of each word to become a keyword in the article.

<div align=center><img src="./these.images/算法1.png" alt="算法1" style="zoom:33%;" /></div>  

| Keyword | weight              |
| ------- | ------------------- |
| 金山    | 0.33799462145286435 |
| 疫情    | 0.2844514062364321  |
| 防控    | 0.2538705144310553  |
| 11      | 0.24029683422914572 |
| 问需    | 0.1802226256718593  |
| 0591    | 0.1802226256718593  |
| 街道    | 0.17984414613517588 |
| 居民    | 0.14645312820025128 |
| 配合    | 0.12727199977025128 |
| 2022    | 0.12014841711457286 |
| 12      | 0.12014841711457286 |
| 38      | 0.12014841711457286 |
| 战役    | 0.12014841711457286 |
| 醉美    | 0.12014841711457286 |
| 仓山    | 0.12014841711457286 |
| 朋友    | 0.11752981875055277 |
| 全体    | 0.10694243655648242 |
| 防疫    | 0.10018946753015076 |
| 大家    | 0.09791958812201006 |

However, there is a main **problem**: the corpus used in the inverse document frequency of TF-IDF is not fully applicable to this project, and **Arabic numerals** may appear, so we do not use this algorithm.

#### TextRank

We actually use the [**TextRank**](https://www.researchgate.net/publication/200042361_TextRank_Bringing_Order_into_Text) algorithm. It is based on [graph theory](https://www.geeksforgeeks.org/mathematics-graph-theory-basics-set-1/). Graphs contain **vertices** and **edges**, which represent **lexical units** and **relationships** between different words in an article. Below is an example graph of words and their relationships.

<div align=center><img src="./these.images/image-20230111160828029.png" alt="image-20230111160828029" style="zoom:20%;" /></div>  

The core concept of **TextRank** algorithm is "**voting**" or "**recommending**". **Notice** that the importance of the vertex determines the importance of the edge connected to the vertex. Here is a brief explanation for TextRank algorithms.  
<div align=center><img src="./these.images/公式2.png" alt="公式2" style="zoom:100%;" /></div>  
We use relationship between connected vertices and calculate them iteratively with a set of initial values assigned to each vertex. Also note that the initial value does not influence the ultimate result of the algorithm.

TextRank mainly including the following steps.

+ To add the determined keyword as a vertex to the graph according to the lexical unit, and then connect the vertex according to the **relationship** between the two words. 

+ Perform iterative calculation until convergence
+ Rank the vertices according to the vertex score to obtain the ranking of keywords

Also, using the [article](https://mp.weixin.qq.com/s/JqxjRVQxflbb8J-oXM5nFw) mentioned above, we can have the TextRank keywords for the article.

| Keyword | weight              |
| ------- | ------------------- |
| 疫情    | 1.0                 |
| 防控    | 0.8950253329225641  |
| 配合    | 0.5923748399190941  |
| 街道    | 0.5009298772676438  |
| 居民    | 0.48506052741948785 |
| 大家    | 0.45676645755077533 |
| 朋友    | 0.41273400068681554 |
| 全体    | 0.3761215829871847  |
| 仓山    | 0.2987042528549764  |
| 工作    | 0.290975951924855   |
| 做好    | 0.28282564380346414 |
| 防疫    | 0.2776601659796551  |
| 得益于  | 0.2725217800096617  |
| 热线    | 0.26051725373979934 |


In our project, we collected **26** tweets from the subscription from 问需金山, and the time range was September to December.

### Synonym

However, there is a **problem** with the above algorithms, that they cannot query **synonyms**. Actually there are many words exhibiting the **same** meaning in the output keywords. So we need to deal with the synonym problem.

<div align=center><img src="./these.images/WechatIMG72.JPEG" alt="WechatIMG72" style="zoom:33%;" /></div>  

Therefore, we have introduced [**cnsyn**](https://gitee.com/vencen/Chinese-Synonyms) to build synonym thesaurus using **Wikipedia** and **Chinese synonym dictionary**. When the user enters the query word, search the synonym of the word in the inverted index according to the word, and return the synonym of the input word.



## Algorithm Implementation

[This](https://github.com/changyang21/stat3060-website/blob/main/main.py) is the whole source code in @github for our project. We use different packages in python to realize different algorithms mentioned before.

### Overview

This **flowchart** below roughly shows the implementation process of the algorithm.

<div align=center><img src="./these.images/image-20230111162903211.png" alt="image-20230111162903211" style="zoom:15%;" /></div>  

And here is the main steps:

+ We need to **preproces** the article first, to use TextRank algorithm to extract keywords of 26 articles and record them in the data structure.

+ We want to implement two searching methods in the project, namely, searching for **article keywords** and searcihng for **full-text content**.

	+ The keywords entered by the user can be used as the **substring** of the article keyword (for example, the user enters *核酸*, and the program search articles containing *做核酸*).
		+ Also, they can serve as the **synonym** (for example, the relationship between "*抗疫* and *防控疫情*).
		+ From the perspective of **relevance**, the former will have **higher priority** in ranking than the latter.

	+ In the content searching method, the algorithm will simply **traverse** all articles and return links to articles containing the words entered by users.

### Results

Here, we present our result for the project.

#### Keyword Searching

The first is **keyword searching**. After entering *防疫* and *阳*, you can see that 2 and 3 articles containing relevant keywords in the database are returned here.

<div align=center><img src="./these.images/image-20230111164335387.png" alt="image-20230111164335387" style="zoom:100%;" /></div>  

However, note that the search for keywords here takes **14 seconds**, which is a long time and is not feasible in practice. This problem will be analyzed later in the paper.

#### Full-Text Content Searching

The second is **full text content searching**. Input the full text of *做核酸* and *水果* to search the whole article. The program will return those web links for the articles.

<div align=center><img src="./these.images/image-20230111164523666.png" alt="image-20230111164523666" style="zoom:100%;" /></div>  

#### Problem Solved

Finally, we have solved the pain points mentioned at the beginning of the report and realize **keyword extraction** and **searching** function in the python program.

For example, this table represents the hot words entered by the user in the chat box of the official account. 

| Rank | Keyword | Input Times |
| ---- | ------- | ----------- |
| 1    | 秒杀    | 608         |
| 2    | 核酸    | 381         |
| 3    | 社区    | 194         |
| 4    | 看病    | 39          |
| 5    | 金山    | 34          |
| 6    | 做      | 32          |
| 7    | 信息    | 31          |
| 8    | 健康    | 39          |
| 9    | 你好    | 26          |
| 10   | 检测    | 26          |

The algorithm can be introduced to **automatically respond** to **any keyword**.



## Analysis
In this part, we will analyze the algorithm performance in detail.

### Keyword Extraction Precision

Namely, human extraction keywords will serve as a **standard answer** for testing our algorithm (Even though this is a fairly **subjective** answer). The formula will be listed below  
<div align=center><img src="./these.images/公式3.png" alt="公式3" style="zoom:100%;" /></div>  
Compare the keywords extracted by manpower and algorithm. Note that we four group members extract four sets of keywords and then we take intersection of our keyword sets.

The figure below shows the keywords extracted by our four members for a subscription article and ranked in descending order by the number of overlaps:

<div align=center><img src="./these.images/image-20230111165721064.png" alt="image-20230111165721064" style="zoom:40%;" /></div>  

while this figure below shows the keywords retrieved by the **algorithm** and ranked by the relevance:

<div align=center><img src="./these.images/image-20230111165835777.png" alt="image-20230111165835777" style="zoom:50%;" /></div>  

We filtered the keywords given by the algorithm with the manually selected keywords as the criteria. Then, we can find a total of **6 keywords** that are in line with each other, which is indicated by the red bar chart on the right.

In fact, the algorithm gave a total of 20 keywords, because the latter ones did not overlap and had low relevance, they were not placed in the chart. After a rough calculation, we can conclude that this algorithm has an accuracy of 30%.  
<div align=center><img src="./these.images/公式4.png" alt="公式4" style="zoom:100%;" /></div>  
At the same time, we found that the highest keyword accuracy obtained by the TextRank algorithm was 31.2% by searching the relevant literature, which is close to the result of 30% obtained by our algorithm.

<div align=center><img src="./these.images/image-20230111170442240.png" alt="image-20230111170442240" style="zoom:25%;" /></div>  

Moreover, since most of the valid keywords are concentrated in the first 10, we can further **increase the precision** by delimiting the **keyword relevance range**, for example, by limiting the relevance to greater than 0.4. Therefore, I think the precision of the algorithm meets the requirement of use.

### Efficiency for Keyword Searching

In terms of the efficiency of the algorithm, we found that the time for each keyword searching is more than 10 seconds. This is because our search for keywords includes **synonym searching**, and the algorithm needs to **cross-reference** the synonyms of the search terms with the synonyms of the extracted keywords, each comparison requiring re-searching for synonyms.

<div align=center><img src="./these.images/image-20230111170609160.png" alt="image-20230111170609160" style="zoom:40%;" /></div>  

Furthermore, the search for synonyms requires access to multiple **web resources**, so the overall efficiency is even lower.

For this, our proposed solution is to build a **local thesaurus** and put in advance the synonyms of all article keywords, as well as the synonyms, just like the memory structure of the computer, which put things that may be used into the main memory and cache in advance to increase efficiency.

<div align=center><img src="./these.images/image-20230111171002169.png" alt="image-20230111171002169" style="zoom:40%;" /></div>  

## Reference

[1] Ramos, Juan. (2003). Using TF-IDF to determine word relevance in document queries.  
[2] Mihalcea, Rada & Tarau, Paul. (2004). TextRank: Bringing Order into Text..  
[3] [Chinese Synonyms 中文同义词查询工具包 Chinese Synonyms for Natural Language Processing and Understanding.](https://gitee.com/vencen/Chinese-Synonyms)



## Related Materials

This is our [Source Code](https://github.com/changyang21/stat3060-website/blob/main/main.py) and [Slides](https://www.kdocs.cn/l/cav3jlHlzYWw?from=docs) of the whole program. You can refer to it and contact us.  
Cheng Zheng alex_zc@sjtu.edu.cn  
Jinhua Sun dt8281_x-.15@sjtu.edu.cn  
Yuliang Zhang blu_26@sjtu.edu.cn  
Yejia Liu wendyliuyejia@sjtu.edu.cn  
