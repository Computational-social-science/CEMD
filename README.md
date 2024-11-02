# CEMD

Existing misinformation detection benchmark datasets (e.g., COVMIS and LIAR2) are limited by their reliance on fact-checking labels that are prone to factual inaccuracies due to cognitive constraints of fact-checkers and outdated labels. Prior real-time misinformation detection tasks have been hindered by the dual problems of label redundancy and cold start. To this end, we propose a novel Cyclic Evidence-based Misinformation Detection (CEMD) framework, which incorporates two core mechanisms: (i) a Retrieval Augmented Generation (RAG) pipeline that accesses the latest external knowledge to augment insufficient prior knowledge, and (ii) a cyclic evidence-bootstrapping mechanism that mitigates label redundancy and cold start.

## Evaluation of RAG

### Retrieval
<img src="1. Evaluation of RAG/assets/umap.svg" style="zoom: 75%;" />

Visualizing claim-context relevance with UMAP.

<br/><br/>

<img src="1. Evaluation of RAG/assets/t-sne.svg" style="zoom: 75%;" />

Visualizing claim-context relevance with t-SNE.

### generation
<img src="1. Evaluation of RAG/assets/ragas.svg" style="zoom: 75%;" />

Comparison of model performance in faithfulness and answer correctness metrics. 

## Evaluation of Classification
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

<img src="2. Evaluation of Classification/assets/human_baseline.svg" style="zoom: 75%;" />

Comparative analysis of LLM combinations and fine-tuning strategies with human baseline (the center).

## Performance
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
Binary classification experiments using all the data from the dataset. XY denotes the portion of the data initially labeled as X that is recategorized as Y. T: True. PT: Partly True. F: False. MT: Mostly True. HT: Half True. BT: Barely True. PF: Pants on Fire.

## Ablation Study

| Method            | Acc              | F1               | P                | R                |
| :---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| No RAG + No SFT   | 0.7902           | 0.7632           | 0.7548           | 0.8339           |
| RAG (RA) + No SFT | 0.8828           | 0.8465           | 0.8376           | 0.8571           |
| RAG (OS) + No SFT | 0.9107           | 0.8830           | 0.8731           | 0.8947           |
| No RAG + SFT      | 0.9770           | 0.9686           | 0.9754           | 0.9623           |
| RAG (RA) + SFT    | 0.9852           | 0.9797           | 0.9892           | 0.9711           |
| RAG (OS) + SFT    | **0.9885**       | **0.9844**       | **0.9901**       | **0.9789**       |
Ablation Study. The dataset used is COVMIS2, where RA stands for related articles, and OS represents online search.

## Real-time Detection
<img src="6. Real-time Detection/assets/timeline2.svg" style="zoom: 75%;" />

Misinformation detection with regular 24-hour updates
