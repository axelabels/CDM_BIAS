This repository contains the code to reproduce the results of our paper "Dealing with Expert Bias in Collective Decision-Making". If you use this code in your own research, please cite this paper:

```
@article{abels2021dealing,
  title={Dealing with Expert Bias in Collective Decision-Making},
  author={Abels, Axel and Lenaerts, Tom and Trianni, Vito and Now{\'e}, Ann},
  journal={arXiv preprint arXiv:2106.13539},
  year={2021}
}
```
---------------------------------------
#### Abstract

Quite some real-world problems can be formulated as decision-making problems wherein one must repeatedly make an appropriate choice from a set of alternatives. Expert judgements, whether human or artificial, can help in taking correct decisions, especially when exploration of alternative solutions is costly. As expert opinions might deviate, the problem of finding the right alternative can be approached as a collective decision making problem (CDM). 
Current state-of-the-art approaches to solve CDM are limited by the quality of the best expert in the group, and perform poorly if experts are not qualified or if they are overly biased, thus potentially derailing the decision-making process. In this paper, we propose a new algorithmic approach based on contextual multi-armed bandit problems (CMAB) to identify and counteract such biased expertises. We explore homogeneous, heterogeneous and polarised expert groups and show that this approach is able to effectively exploit the collective expertise, outperforming state-of-the-art methods, especially when the quality of the provided expertise degrades. Our novel CMAB-inspired approach achieves a higher final performance and does so while converging more rapidly than previous adaptive algorithms, especially when heterogeneous expertise is readily available.
