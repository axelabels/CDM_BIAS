[![DOI](https://zenodo.org/badge/429481250.svg)](https://zenodo.org/badge/latestdoi/429481250)


This repository contains the code to reproduce the results of our paper "Dealing with Expert Bias in Collective Decision-Making". If you use this code in your own research, please cite this paper:

```
@article{abels2023dealing,
  title={Dealing with expert bias in collective decision-making},
  author={Abels, Axel and Lenaerts, Tom and Trianni, Vito and Now{\'e}, Ann},
  journal={Artificial Intelligence},
  volume={320},
  pages={103921},
  year={2023},
  publisher={Elsevier}
}
```
---------------------------------------
#### Abstract

Quite some real-world problems can be formulated as decision-making problems wherein one must repeatedly make an appropriate choice from a set of alternatives. Multiple expert judgments, whether human or artificial, can help in taking correct decisions, especially when exploration of alternative solutions is costly. As expert opinions might deviate, the problem of finding the right alternative can be approached as a collective decision making problem (CDM) via aggregation of independent judgments.
Current state-of-the-art approaches focus on efficiently finding the optimal expert, and thus perform poorly if all experts are not qualified or if they display consistent biases, thereby potentially derailing the decision-making process. In this paper, we propose a new algorithmic approach based on contextual multi-armed bandit problems (CMAB) to identify and counteract such biased expertise. We explore homogeneous, heterogeneous and polarized expert groups and show that this approach is able to effectively exploit the collective expertise, outperforming state-of-the-art methods, especially when the quality of the provided expertise degrades. Our novel CMAB-inspired approach achieves a higher final performance and does so while converging more rapidly than previous adaptive algorithms.
