# CEMD

Prior real-time misinformation detection tasks have been hindered by the dual problems of label redundancy and cold start. To this end, we propose a novel Cyclic Evidence-based Misinformation Detection (CEMD) framework, which incorporates two core mechanisms: (i) a Retrieval Augmented Generation (RAG) pipeline that accesses the latest external knowledge to augment insufficient prior knowledge, and (ii) a cyclic evidence-bootstrapping mechanism that mitigates label redundancy and cold start.

## Dataset
Existing misinformation detection benchmark datasets (e.g., COVMIS and LIAR2) are limited by their reliance on fact-checking labels that are prone to factual inaccuracies due to cognitive constraints of fact-checkers and outdated labels. 

[COVMIS](https://github.com/caryou/COVMIS): COVMIS was constructed to support the misinformation identification approach that mimics the act of fact checking by human for truth labelling. COVMIS is collected from November 2019 to March 2021, this dataset contains 14,384 claims (statements), 134,320 related articles, and many features associated with the claims such as claimants, news sources, dates, truth labels (true, partly true or false) and justifications for the truth labels.

[LIAR2](https://github.com/chengxuphd/liar2): The LIAR2 dataset is an upgrade of the LIAR dataset, which inherits the ideas of the LIAR dataset, refines the details and architecture, and expands the size of the dataset to make it more responsive to the needs of fake news detection tasks.

<table>
    <tr>
        <th>Dataset</th>
        <th>Label</th>
        <th>Number</th>
    </tr>
    <tr>
        <td rowspan="3">COVMIS2</td>
        <td>T</td>
        <td>1998</td>
    </tr>
    <tr>
        <td>PT</td>
        <td>2192</td>
    </tr>
    <tr>
        <td>F</td>
        <td>10038</td>
    </tr>
    <tr>
        <td rowspan="6">LIAR2</td>
        <td>T</td>
        <td>2585</td>
    </tr>
    <tr>
        <td>MT</td>
        <td>3429</td>
    </tr>
    <tr>
        <td>HT</td>
        <td>3709</td>
    </tr>
    <tr>
        <td>BT</td>
        <td>3603</td>
    </tr>
    <tr>
        <td>F</td>
        <td>6605</td>
    </tr>
    <tr>
        <td>PF</td>
        <td>3031</td>
    </tr>
    </tr>
</table>

The composition of the dataset. T: *True*. PT: *Partly True*. F: *False*. MT: *Mostly True*. HT: *Half True*. BT: *Barely True*. PF: *Pants on Fire*.

## Evaluation of RAG

### Retrieval

We use *t*-SNE and UMAP to reduce the dimensionality of the text embedding of each claim and the corresponding retrieved contexts for the clustering analysis.

Embedding model: [mxbai-embed-large-v1](https://www.mixedbread.ai/docs/embeddings/mxbai-embed-large-v1)

Dimensionality reduction: 1024 to 2
<br/>

For the clustering analysis using UMAP projections, the features of retrieved contexts tend to group together in neighborhoods of each claim. We find that the average distance for the top 95% ranges from 0 to 0.2294 (data is normalized).

<img src="1. Evaluation of RAG/assets/umap.svg" style="zoom: 45%;" />

Visualizing claim-context relevance with UMAP.

<br/>
For the clustering analysis use t-SNE projections, the features of retrieved contexts tend to group together in neighborhoods of each claim. We find that the average distance for the top 95% ranges from 0 to 0.2601 (data is normalized).

<img src="1. Evaluation of RAG/assets/t-sne.svg" style="zoom: 45%;" />

Visualizing claim-context relevance with *t*-SNE.

### Generation

[Ragas](https://docs.ragas.io/en/stable/) is a library that provides tools to supercharge the evaluation of Large Language Model (LLM) applications. We evaluated the generative performance of the RAG pipeline using Ragas, selecting the top-performing LLM for subsequent classification experiments.

`Faithfulness` metric measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The faithfulness score in our task is given by:

$$
\text{Faithfulness score} = \frac{\left| \text{Number of statements in the generated answer that can be inferred from the contexts} \right|}{\left| \text{Total number of statements in the generated answer} \right|}
$$

> **Example**
>
> Question: Please judge the correctness of the CLAIM "COVID-19 is an airborne disease" based on the available information.
>
> Contexts: Under experimental conditions, researchers found that the COVID-19 virus stayed viable in the air for three hours. The researchers estimate that in most real-world situations, the virus would remain suspended in the air for about 30 minutes, before settling onto surfaces. ... On 23 December, the World Health Organization (WHO) uttered the one word it had previously seemed incapable of applying to the virus SARS-CoV-2: ‘airborne’.
>
> Answer: COVID-19 can remain viable in the air for extended periods (up to six hours in experimental conditions and 30 minutes in real-world situations). Given the consistency and credibility of these sources, it is reasonable to conclude that the CLAIM "COVID-19 is an airborne disease" is TRUE.

* **Step 1:** Break the generated answer into individual statements.
  * Statements:
    * Statement 1: COVID-19 can remain viable in the air for extended periods, up to six hours in experimental conditions and 30 minutes in real-world situations.
    * Statement 2: Given the consistency and credibility of these sources, it is reasonable to conclude that the CLAIM 'COVID-19 is an airborne disease' is TRUE.
* **Step 2:** For each of the generated statements, verify if it can be inferred from the given context.
  - Statement 1: No
  - Statement 2: Yes
* **Step 3:** Use the formula depicted above to calculate faithfulness: $\text{Faithfulness score} = \frac{1}{2} = 0.5$

<br/><br/>
The assessment of `Answer Correctness` involves gauging the accuracy of the generated answer when compared to the ground truth.

Answer correctness is computed as the sum of factual correctness and the semantic similarity between the given answer and the ground truth.

Factual correctness quantifies the factual overlap between the generated answer and the ground truth answer. This is done using the concepts of:

- TP (True Positive): Facts or statements that are present in both the ground truth and the generated answer.
- FP (False Positive): Facts or statements that are present in the generated answer but not in the ground truth.
- FN (False Negative): Facts or statements that are present in the ground truth but not in the generated answer.

Now, we can use the formula for the F1 score to quantify correctness based on the number of statements in each of these lists:

$$
\text{Factual correctness} = \frac{2 \cdot \left| \text{TP} \right|}{2 \cdot \left| \text{TP} \right| + \left| \text{FP} \right| + \left| \text{FN} \right|}
$$

Next, we calculate the semantic similarity between the generated answer and the ground truth. Once we have the semantic similarity, we take a weighted (0.75 and 0.25) average of the factual correctness and the semantic similarity calculated above to arrive at the final score.

$$
\text{Answer correctness score} = 0.75 \cdot \text{Factual correctness} + 0.25 \cdot \text{Semantic similarity}
$$

> **Example**
>
> Ground truth: The CLAIM "COVID-19 is an airborne disease" is TRUE.
>
> Answer: Given the consistency and credibility of these sources, it is reasonable to conclude that the CLAIM "COVID-19 is an airborne disease" is TRUE.

* **Step 1:** Calculate TP, FP and FN using the above rules.
  * $\left| \text{TP} \right|$ = 1
  * $\left| \text{FP} \right|$ = 0
  * $\left| \text{FN} \right|$ = 0
* **Step 2:** Use the formula depicted above to calculate factual correctness: $\text{Factual correctness} = \frac{2 \cdot 1}{2 \cdot 1 + 0 + 0} = 1$
* **Step 3:** Calculate the semantic similarity between the generated answer and the ground truth: $\text{Semantic similarity} = \theta$
* **Step 4:** Use the formula depicted above to calculate answer correctness score: $\text{Answer correctness score} = 0.75 + 0.25\theta$

<br/>
<img src="1. Evaluation of RAG/assets/ragas.svg" style="zoom: 35%;" />

Comparison of model performance in faithfulness and answer correctness metrics.

## Evaluation of Binary Classification

We conduct multiple binary classification experiments using data labeled as True and False from COVMIS2, combining various LLMs and fine-tuning strategies. Through extensive experimentation, we identified the optimal configuration for our CEMD framework. The most effective combination, named as `CEMDo`, includes the Llama-3-70B-Instruct LLM in the RAG pipeline, the Llama-3-8B-Instruct LLM for classification, and the DoRA fine-tuning strategy.
<br/>

<table>
    <tr>
        <th>Model</th>
        <th>Strategy</th>
        <th>Acc</th>
        <th>F1</th>
        <th>P</th>
        <th>R</th>
    </tr>
    <tr>
        <td rowspan="5">OpenChat-3.5-0106</td>
        <td>LoRA</td>
        <td>0.9836</td>
        <td>0.9775</td>
        <td>0.9857</td>
        <td>0.9700</td>
    </tr>
    <tr>
        <td>LoRA+</td>
        <td>0.9869</td>
        <td>0.9821</td>
        <td>0.9890</td>
        <td>0.9756</td>
    </tr>
    <tr>
        <td>VeRA</td>
        <td>0.9844</td>
        <td>0.9787</td>
        <td>0.9862</td>
        <td>0.9717</td>
    </tr>
    <tr>
        <td>rsLoRA</td>
        <td>0.9836</td>
        <td>0.9776</td>
        <td>0.9845</td>
        <td>0.9712</td>
    </tr>
    <tr>
        <td>DoRA</td>
        <td>0.9877</td>
        <td>0.9833</td>
        <td>0.9884</td>
        <td>0.9784</td>
    </tr>
    <tr>
        <td rowspan="5">Mistral-7B-Instruct-v0.3</td>
        <td>LoRA</td>
        <td>0.9869</td>
        <td>0.9821</td>
        <td>0.9890</td>
        <td>0.9756</td>
    </tr>
    <tr>
        <td>LoRA+</td>
        <td>0.9877</td>
        <td>0.9833</td>
        <td>0.9884</td>
        <td>0.9784</td>
    </tr>
    <tr>
        <td>VeRA</td>
        <td>0.9852</td>
        <td>0.9798</td>
        <td>0.9867</td>
        <td>0.9734</td>
    </tr>
    <tr>
        <td>rsLoRA</td>
        <td>0.9877</td>
        <td>0.9833</td>
        <td>0.9884</td>
        <td>0.9784</td>
    </tr>
    <tr>
        <td>DoRA</td>
        <td>0.9877</td>
        <td>0.9833</td>
        <td>0.9884</td>
        <td>0.9784</td>
    </tr>
    <tr>
        <td rowspan="5">Llama-3-8B-Instruct</td>
        <td>LoRA</td>
        <td>0.9869</td>
        <td>0.9821</td>
        <td>0.9878</td>
        <td>0.9767</td>
    </tr>
    <tr>
        <td>LoRA+</td>
        <td>0.9877</td>
        <td>0.9832</td>
        <td>0.9896</td>
        <td>0.9772</td>
    </tr>
    <tr>
        <td>VeRA</td>
        <td>0.9869</td>
        <td>0.9821</td>
        <td>0.9878</td>
        <td>0.9767</td>
    </tr>
    <tr>
        <td>rsLoRA</td>
        <td>0.9877</td>
        <td>0.9833</td>
        <td>0.9884</td>
        <td>0.9784</td>
    </tr>
    <tr>
        <td>DoRA</td>
        <td><b>0.9885</b></td>
        <td><b>0.9844</b></td>
        <td><b>0.9901</b></td>
        <td><b>0.9789</b></td>
    </tr>
</table>
Classification results of different LLMs and fine-tuning strategies.

<br/><br/>

<img src="2. Evaluation of Classification and Performance/assets/human_baseline.svg" style="zoom: 75%;" />

Comparative analysis of LLM combinations and fine-tuning strategies with human baseline (the center).

## Performance

We also conduct experiments using the LIAR2 dataset and compare the performance of [FDHN](https://github.com/chengxuphd/FDHN) with that of CEMDo. It is evident that CEMDo outperforms FDHN on both COVMIS2 and LIAR2.

<table>
  <tr>
    <th>Dataset</th>
    <th>Method</th>
    <th>Acc</th>
    <th>F1</th>
    <th>P</th>
    <th>R</th>
  </tr>
  <tr>
    <td rowspan="2">COVMIS2</td>
    <td>FDHN</td>
    <td>0.9459</td>
    <td>0.9269</td>
    <td>0.9279</td>
    <td>0.9259</td>
  </tr>
  <tr>
    <td>CEMDo</td>
    <td><b>0.9885</b></td>
    <td><b>0.9844</b></td>
    <td><b>0.9901</b></td>
    <td><b>0.9789</b></td>
  </tr>
  <tr>
    <td rowspan="2">LIAR2</td>
    <td>FDHN</td>
    <td>0.8183</td>
    <td>0.6465</td>
    <td>0.7501</td>
    <td>0.6236</td>
  </tr>
  <tr>
    <td>CEMDo</td>
    <td><b>0.9378</b></td>
    <td><b>0.9074</b></td>
    <td><b>0.9052</b></td>
    <td><b>0.9097</b></td>
  </tr>
</table>
Performance of FDHN and CEMDo on two different datasets.

## Data Recategorization

We recategorize data with redundant labels using CEMDo fine-tuned with data labeled as *TRUE* or *FALSE*. Subsequently, we conduct binary classification experiments with all data from each dataset using CEMDo.

<table>
    <tr>
        <th>Dataset</th>
        <th>Label</th>
        <th>Number</th>
    </tr>
    <tr>
        <td rowspan="4">COVMIS2</td>
        <td>T</td>
        <td>2989</td>
    </tr>
    <tr>
        <td>PT<sub>TRUE</sub></td>
        <td>426</td>
    </tr>
    <tr>
        <td>F</td>
        <td>9203</td>
    </tr>
    <tr>
        <td>PT<sub>FALSE</sub></td>
        <td>1766</td>
    </tr>
    <tr>
        <td rowspan="9">LIAR2</td>
        <td>T</td>
        <td>2585</td>
    </tr>
    <tr>
        <td>MT<sub>TRUE</sub></td>
        <td>2500</td>
    </tr>
    <tr>
        <td>HT<sub>TRUE</sub></td>
        <td>1361</td>
    </tr>
    <tr>
        <td>BT<sub>TRUE</sub></td>
        <td>583</td>
    </tr>
    <tr>
        <td>F</td>
        <td>6605</td>
    </tr>
    <tr>
        <td>PF</td>
        <td>3031</td>
    </tr>
    <tr>
        <td>MT<sub>FALSE</sub></td>
        <td>929</td>
    </tr>
    <tr>
        <td>HT<sub>FALSE</sub></td>
        <td>2348</td>
    </tr>
    <tr>
        <td>BT<sub>FALSE</sub></td>
        <td>3020</td>
    </tr>
</table>

The composition of the dataset after recategorization. T: *True*. PT: *Partly True*. F: *False*. MT: *Mostly True*. HT: *Half True*. BT: *Barely True*. PF: *Pants on Fire*.

<br/>

<table>
  <tr>
    <th>Dataset</th>
    <th>Labeling method</th>
    <th>Acc</th>
    <th>F1</th>
    <th>P</th>
    <th>R</th>
  </tr>
  <tr>
    <td>COVMIS2</td>
    <td>TRUE: T, PT<sub>TRUE</sub>; <br> FALSE: F, PT<sub>FALSE</sub>;</td>
    <td>0.9882</td>
    <td>0.9828</td>
    <td>0.9856</td>
    <td>0.9801</td>
  </tr>
  <tr>
    <td rowspan="2">LIAR2</td>
    <td>TRUE: T, MT, HT; <br> FALSE: BT, F, PF;</td>
    <td>0.8641</td>
    <td>0.8621</td>
    <td>0.8602</td>
    <td>0.8659</td>
  </tr>
  <tr>
      <td>TRUE: T, MT<sub>TRUE</sub>, HT<sub>TRUE</sub>, BT<sub>TRUE</sub>; <br> FALSE: F, PF, MT<sub>FALSE</sub>, HT<sub>FALSE</sub>, BT<sub>FALSE</sub>;</td>
    <td>0.9325</td>
    <td>0.9218</td>
    <td>0.9210</td>
    <td>0.9226</td>
  </tr>
</table>

Binary classification experiments using all the data from the dataset. X<sub>Y</sub> denotes the portion of the data initially labeled as *X* that is recategorized as *Y*. T: *True*. PT: *Partly True*. F: *False*. MT: *Mostly True*. HT: *Half True*. BT: *Barely True*. PF: *Pants on Fire*.

## Ablation Study

Explore the differences between 6 strategies: the RAG pipeline combined with online search, the RAG pipeline using a local knowledge base, the strategy without using a RAG pipeline and each of these strategies combined with the SFT. We use COVMIS2 as the dataset for the comparative experiments, with related articles serving as the local knowledge base.

The results show that using RAG and SFT can improve the misinformation detection.

| Method            | Acc              | F1               | P                | R                |
| :---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| No RAG + No SFT   | 0.7902           | 0.7632           | 0.7548           | 0.8339           |
| RAG (RA) + No SFT | 0.8828           | 0.8465           | 0.8376           | 0.8571           |
| RAG (OS) + No SFT | 0.9107           | 0.8830           | 0.8731           | 0.8947           |
| No RAG + SFT      | 0.9770           | 0.9686           | 0.9754           | 0.9623           |
| RAG (RA) + SFT    | 0.9852           | 0.9797           | 0.9892           | 0.9711           |
| RAG (OS) + SFT    | **0.9885** | **0.9844** | **0.9901** | **0.9789** |

Ablation Study. The dataset used is COVMIS2, where RA stands for related articles, and OS represents online search. The model in the RAG pipeline is Llama-3-70B-Instruct. For the SFT, we employ Llama-3-8B-Instruct, fine-tuned with DoRA.

## Real-time Detection

The CEMD framework harnesses external knowledge to augment the initial prior knowledge with regular 24-hour updates, thereby mitigating label redundancy and addressing the cold start problem.

<img src="5. Real-time Detection/assets/timeline2.svg" style="zoom: 75%;" />

Misinformation detection with regular 24-hour updates.

## Citing this work
The relevant paper is currently under review, during which time this repository is private. Once it goes public, a bibtex reference will be provided here.
