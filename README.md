---

<div align="center">      

# Borrowing Knowledge From Pre-trained Language Model: A New Data-efficient Visual Learning Paradigm

Wenxuan Ma, [Shuang Li](https://shuangli.xyz), Jinming Zhang, [Chi Harold Liu](https://scholar.google.com/citations?user=3IgFTEkAAAAJ&hl=en), Jingxuan Kang, [Yulin Wang](https://www.rainforest-wang.cool/), and Gao Huang

</div>

Official implementation of our ICCV 2023 paper (BorLan). 

## Paradigm Introduction

BorLan is a simple data-efficient learning paradigm that includes three parts:

1. Obtain text embedding of task concepts via pre-trained language model (PLM). (This part can be conducted before the visual training once and for all for a given dataset.)
2. Main task loss (i.e., CrossEntropy)
3. Distribution alignment loss that leverages text embedding space to promote data-efficient visual training.


<img src="resources\borlan_paradigm.png" width=100% height=100%>

## Training 

### Step 1: Obtain text embedding of concepts via PLM.

Run the following command to obtain text embeddings.

You need to modify the following things in the code:

- classnames: List
- save_name: str 

```python
# Bert-Large
python text_features/text_embedding.py

# GPT-2
python text_features/text_embedding_gpt.py

# CLIP ViT-Large
python text_features/text_embedding_clip.py
```

### Step 2: Linguistic knowledge guided vision model training.

Run the following command for Semi-Supervised Learning tasks:

```python
sh run.sh
```

## Acknowledgement

This repository borrows codes from the following repos. Many thanks to the authors for their great work.

Self-Tuning: https://github.com/thuml/Self-Tuning

CoOp:  https://github.com/KaiyangZhou/CoOp

## Citation

If you find this project useful, please consider citing:

```
@inproceedings{ma2023borrowing,
  title={Borrowing Knowledge From Pre-trained Language Model: A New Data-efficient Visual Learning Paradigm},
  author={Ma, Wenxuan and Li, Shuang and Zhang, Jinming and Liu, Chi Harold and Kang, Jingxuan and Wang, Yulin and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  year={2023}
}
```

## Contact

If you have any questions about our code, feel free to contact us or describe your problem in Issues.

Email address: wenxuanma@bit.edu.cn.

<div align="right">
<b><a href="#overview">↥</a></b>
</div>
