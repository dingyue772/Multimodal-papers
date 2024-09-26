# Awesome Modality Alignment
Modality Alignment refers to the process of ensuring that different modalities are appropriately aligned. Specifically, in Large Vision-Language Models (LVLMs), the goal is to align visual tokens with the embedding space of the large language model.

In the LVLM's community, there're some methods to bridge the gap between visual and textual representations.

## Measure modality alignment

- **AC score**: Law of Vision Representation in MLLMs(Aug 29, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.16357) [![Star](https://img.shields.io/github/stars/bronyayang/law_of_vision_representation_in_mllms.svg?style=social&label=Star)](https://github.com/bronyayang/law_of_vision_representation_in_mllms)
  - analyze cross-modality alignment and vision correspondence of vision representation and their influence on LVLMs' performance
- **The Platonic Representation Hypothesis**(May 13, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.07987) [![Star](https://img.shields.io/github/stars/minyoungg/platonic-rep.svg?style=social&label=Star)](https://github.com/minyoungg/platonic-rep)
  - the way different neural networks represent data is becoming increasingly uniform
    - Convergence of Cross-Modal Data Representations: As model sizes increase, the similarity between visual models and language models in representing data also increases.
    - Drivers of Representation Convergence: The universality of tasks, the increase in model capacity, and reduced inductive bias
- **Mind the Gap**: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning(May 3, 2022) **NeurIPS 2022**
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.02053) [![Star](https://img.shields.io/github/stars/weixin-liang/modality-gap.svg?style=social&label=Star)](https://github.com/weixin-liang/modality-gap)
  -  The phenomenon of modal representation differences is demonstrated, and it is proven that this phenomenon is universally present across different data modalities and neural network architectures.
    - General Inductive Bias of Deep Neural Networks
    - Non-linear Activation Functions in Models Effectively Promote the Emergence of Embedding Cones
    - Different Random Initializations Create Different Embedding Cones
    - Preservation of Contrastive Learning Objectives


## Enhance modality alignment
- **SEA**: SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration in MLLMs(Aug. 21, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.11813)
  - follow the pretraining-finetuning paradigm of LLaVA but **add another contrastive loss** $L_a$ in the pretraing stage alongside the next token prediction loss $L_g$, resulting the optimization loss $L=L_g+\lambda L_a$. 
  - The contrastive training is a supervised process. With the CLIP model(the vision encoder $f$ and the text encoder $h$), SEA extracts the most relevant words $S_i$ from a predefined word list for every image patch $v_i$ . 
  - In the similarity calculation, we can gain a pair of {relevant word  $s_u$  and the similarity score $w_u$​}, thus a similarity-weighted sampling method is used for imaga patch-word choosing. 
- **VLAP**: Bridging Vision and Language Spaces with Assignment Prediction(Apr. 15, 2024) **ICRL2024**
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.09632) [![Star](https://img.shields.io/github/stars/park-jungin/vlap.svg?style=social&label=Star)](https://github.com/park-jungin/vlap)
- **V2T Tokenizer**: Beyond Text: Frozen Large Language Models in Visual Signal Comprehension(Mar 12, 2024)
  -  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.07874) [![Star](https://img.shields.io/github/stars/zh460045050/v2l-tokenizer.svg?style=social&label=Star)](https://github.com/zh460045050/v2l-tokenizer)
- **SoM prompting**: List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs(Apr 25, 2024)
  -  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.16375)  [![Star](https://img.shields.io/github/stars/zzxslp/som-llava.svg?style=social&label=Star)](https://github.com/zzxslp/som-llava)
- **AlignGPT**: AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability(May 23, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.14129)  [![Star](https://img.shields.io/github/stars/AlignGPT-VL/AlignGPT.svg?style=social&label=Star)](https://github.com/AlignGPT-VL/AlignGPT)
- **Visual Prompting**: Rethinking Visual Prompting for Multimodal Large Language Models with External Knowledge(Jul 5, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.04681) 
- **a visual embedding highway module**: X-VILA: Cross-Modality Alignment for Large Language Model(May 29, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.19335)
- **LexVLA**: Unified Lexical Representation for Interpretable Visual-Language Alignment(Jul 25, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.17827)
- Vision Transformers Need Registers(Sep 28, 2024, ICLR2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.16588)
- **geometry-aware**: Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence(Nov 28, 2023)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.17034) [![Star](https://img.shields.io/github/stars/Junyi42/geoaware-sc.svg?style=social&label=Star)](https://github.com/Junyi42/geoaware-sc)
- **SIMA**: Enhancing Visual-Language Modality Alignment in Large Vision Language Models via Self-Improvement(May 24, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.15973) [![Star](https://img.shields.io/github/stars/umd-huang-lab/sima.svg?style=social&label=Star)](https://github.com/umd-huang-lab/sima)
- **WCA**: Visual-Text Cross Alignment: Refining the Similarity Score in Vision-Language Models(Jun 5, 2024) **ICML**
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02915) [![Star](https://img.shields.io/github/stars/tmlr-group/wca.svg?style=social&label=Star)](https://github.com/tmlr-group/wca)
- **Multi-Modal Adapter**: Multi-Modal Adapter for Vision-Language Models(Sep 3, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.arxiv.org/abs/2409.02958) 
- **Alt-MoE**: Alt-MoE: Multimodal Alignment via Alternating Optimization of Multi-directional MoE with Unimodal Models(Sep 9, 2024)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.arxiv.org/abs/2409.05929) 
- **Lyrics**: Lyrics: Boosting Fine-grained Language-Vision Alignment and Comprehension via Semantic-aware Visual Objects(Dec 8, 2023)
  - [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.05278) 
