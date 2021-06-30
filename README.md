# MelBERT
This is the official code for the NAACL 2021 paper: [_MelBERT: Metaphor Detection via Contextualized Late Interaction using Metaphorical Identification Theories._](https://www.aclweb.org/anthology/2021.naacl-main.141/).

<!-- The slides can be found [here](). -->

<!-- 
Todo
- script
- slides
- bagging -->


## Dataset

### Detailed statistics of benchmark dataset

| Dataset         | #tokens | %M   | #Seq   | Seq len |
|-----------------|---------|------|--------|---------|
| VUA-18 (train)  | 116,622 | 11.2 |  6,323 |    18.4 |
| VUA-18 (dev)    |  38,628 | 11.6 |  1,550 |    24.9 |
| VUA-18 (test)   |  50,175 | 12.4 |  2,694 |    18.6 |
| VUA-20 (train)  | 160,154 | 12.0 | 12,109 |      15 |
| VUA-20 (test)   |  22,196 | 17.9 |  3,698 |    15.5 |
| VUA-VERB (test) |   5,873 |   30 |  2,694 |    18.6 |
| MOH-X           |     647 | 48.7 |    647 |       8 |
| TroFi           |   3,737 | 43.5 |  3,737 |    28.3 |


We use four well-known public English datasets. The VU Amsterdam Metaphor Corpus (VUA) has been released in metaphor detection shared tasks in 2018 and 2020. We use two versions of VUA datasets, called <b>VUA-18</b> and <b>VUA-20</b>, where VUA-20 is the extension of VUA-18. We split VUA-18 and VUA-20 each for training, validation, and test datasets. VUA-20 includes VUA-18, and VUA-Verb (test) is a subset of VUA-18 (test) and VUA-20 (test). We also use VUA datasets categorized into different POS tags (verb, noun, adjective, and adverb) and genres (news, academic, fiction, and conversation).<br>
We employ <b>MOH-X</b> and <b>TroFi</b> for testing only. 
<br><br>
<!-- Preprocessed Datasets -->

You can get datasets from the link below. 
[https://drive.google.com/file/d/1738aqFObjfcOg2O7knrELmUHulNhoqRz/view?usp=sharing](https://drive.google.com/file/d/1738aqFObjfcOg2O7knrELmUHulNhoqRz/view?usp=sharing)

The datasets are tsv formatted files and the format is as follows.
```
index	label	sentence	POS	w_index
a3m-fragment02 45	0	Design: Crossed lines over the toytown tram: City transport could soon be back on the right track, says Jonathan Glancey	NOUN	0
a3m-fragment02 45	1	Design: Crossed lines over the toytown tram: City transport could soon be back on the right track, says Jonathan Glancey	ADJ	1
a3m-fragment02 45	1	Design: Crossed lines over the toytown tram: City transport could soon be back on the right track, says Jonathan Glancey	NOUN	2
```

You can also get the original datasets from the following links:
<!-- VUA-18 and VUA-20 -->
- VUA-18: [https://github.com/RuiMao1988/Sequential-Metaphor-Identification](https://github.com/RuiMao1988/Sequential-Metaphor-Identification)

- VUA-20: [https://github.com/YU-NLPLab/DeepMet](https://github.com/YU-NLPLab/DeepMet)

<!-- MOH-X -->
- MOH-X: [https://github.com/RuiMao1988/Sequential-Metaphor-Identification](https://github.com/RuiMao1988/Sequential-Metaphor-Identification)

<!-- TroFi -->
- TroFi: [https://github.com/RuiMao1988/Sequential-Metaphor-Identification](https://github.com/RuiMao1988/Sequential-Metaphor-Identification)


<br>

## Basic Usage
- Change the experimental settings in `main_config.cfg`. <br>
- Run `main.py` to train and test models. <br>
- Command line arguments are also acceptable with the same naming in configuration files.

## Running MelBERT

1. Train MelBERT with the specfic huggingface transformer model:<br>
`python main.py --model_type MELBERT --bert_model roberta-base`

2. Test MelBERT with the path of saves file:<br>
`python main.py --model_type MELBERT --bert_model {path of saves file}`

- Using RoBERTa, MelBERT gets about 78.5 and 75.7 F1 scores on the VUA-18 and the VUA-verb set, respectively. Using model bagging techniques, we get about 79.8 and 77.1 F1 scorea on the VUA-18 and VUA-verb set, respectively.
- The argument `task_name` indicates the name of task where 'vua' for VUA datasets and 'trofi' for TroFi and MOH-X datasets. If `task_name` is 'trofi', K-fold is applied for both training and evaluation. 
- The pretrained transformer model can be specified with the argument `bert_model`. The processing of tokenizer may be different for models, so be careful. The work is currently based on RoBERTa-base model.
- The type of model can be specified with the argument `model_type` and the types are as follows.

  ```
  RoBERTa_BASE: BERT_BASE 
  RoBERTa_SEQ: BERT_SEQ
  MelBERT: MELBERT
  MelBERT_MIP: MELBERT_MIP
  MelBERT_SPV: MELBERT_SPV
  ```

<!-- - RoBERTa_BASE: BLT_CLS 
- RoBERTa_SEQ: SEQ_BASE
- MelBERT: CLS_SPV_MIP
- MelBERT_MIP: CLS_MIP
- MelBERT_SPV: CLS_SPV -->

<br>

## Requirements
````
python==3.7
pytorch==1.6
transformers==4.2.2
````

## Citation
Please cite our paper:
```
@inproceedings{DBLP:conf/naacl/ChoiLCPLLL21,
  author    = {Minjin Choi and
               Sunkyung Lee and
               Eunseong Choi and
               Heesoo Park and
               Junhyuk Lee and
               Dongwon Lee and
               Jongwuk Lee},
  title     = {MelBERT: Metaphor Detection via Contextualized Late Interaction using
               Metaphorical Identification Theories},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2021, Online, June 6-11, 2021},
  pages     = {1763--1773},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://www.aclweb.org/anthology/2021.naacl-main.141/},
}
```