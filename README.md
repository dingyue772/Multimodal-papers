This is a repo for organizing papers related to MLLMs. Most papers are from AK's daily papers, and I'll writing reading notes from time to time.

# Daily Papers

1. Arxiv [MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning](https://arxiv.org/abs/2409.20566) (使用MM1的模型架构，采用以数据为中心的模型训练方法，系统探索了整个模型训练生命周期中不同数据混合的影响，推出MoE-based MLLMs, 处理视频理解的MM1.5-Video和处理移动UI理解的MM1.5-UI)
2. NeurIPS [Image Copy Detection for Diffusion Models](https://arxiv.org/abs/2409.19952) (引入了首个专为扩散模型设计的图像复制检测（ICD）模型ICDiff并使用Stable Diffusion v1.5创建了D-Rep数据集) [[Code](https://github.com/WangWenhao0716/PDF-Embedding)]
3. Arxiv [IDEAW: Robust Neural Audio Watermarking with Invertible Dual-Embedding](https://arxiv.org/abs/2409.19627) (为解决当前音频水印技术中存在的容量低、隐蔽性不够和水印定位问题，设计了一个双嵌入水印模型) [[Code](https://github.com/PecholaL/IDEAW)]

# Awesome Modality Alignment
Modality Alignment refers to the process of ensuring that different modalities are appropriately aligned. Specifically, in Large Vision-Language Models (LVLMs), the goal is to align visual tokens with the embedding space of the large language model.

In the LVLM's community, there're some methods to bridge the gap between visual and textual representations.
## 2024
0. Arxiv [Law of Vision Representation in MLLMs](https://arxiv.org/abs/2408.16357) (分析跨模态对齐和视觉表征的视觉对应关系对LVLMs性能的影响) [[Code](https://github.com/bronyayang/law_of_vision_representation_in_mllms)]
1. Arxiv [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) (不同神经网络表示数据的方式正变得越来越统一并进行了相关的分析)
[[Code](https://github.com/minyoungg/platonic-rep)]
2. Arxiv [SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration in MLLMs](https://arxiv.org/abs/2408.11813) (遵循LLaVA的预训练-微调范式，但在预训练阶段增加另一个对比损失$L_a$，与下一个令牌预测损失$L_g$一起，得到新的优化损失$L=L_g+\lambda L_a$)
3. ICRL [VLAP: Bridging Vision and Language Spaces with Assignment Prediction(Apr. 15, 2024](https://arxiv.org/abs/2404.09632) [[Code](https://github.com/park-jungin/vlap)]
4. Arxiv [V2T Tokenizer: Beyond Text: Frozen Large Language Models in Visual Signal Comprehension](https://arxiv.org/abs/2403.07874) [[Code](https://github.com/zh460045050/v2l-tokenizer)]
5. Arxiv [SoM prompting: List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs](https://arxiv.org/abs/2404.16375) [[Code](https://github.com/zzxslp/som-llava)]
6. Arxiv [AlignGPT: AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability](https://arxiv.org/abs/2405.14129) [[Code](https://github.com/AlignGPT-VL/AlignGPT)]
7. [Visual Prompting: Rethinking Visual Prompting for Multimodal Large Language Models with External Knowledge](https://arxiv.org/abs/2407.04681)
8. Arxiv [ X-VILA: Cross-Modality Alignment for Large Language Model](https://arxiv.org/abs/2405.19335)
9. Arxiv [LexVLA: Unified Lexical Representation for Interpretable Visual-Language Alignment](https://arxiv.org/abs/2407.17827)
10. ICLR [ Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
11. Arxiv [SIMA: Enhancing Visual-Language Modality Alignment in Large Vision Language Models via Self-Improvement](https://arxiv.org/abs/2405.15973) [[Code](https://github.com/umd-huang-lab/sima)]
12. ICML [WCA: Visual-Text Cross Alignment: Refining the Similarity Score in Vision-Language Models(Jun 5, 2024)](https://arxiv.org/abs/2406.02915) [[Code](https://github.com/tmlr-group/wca)]
13. Arxiv [Multi-Modal Adapter: Multi-Modal Adapter for Vision-Language Models](https://www.arxiv.org/abs/2409.02958)
14. Arxiv [Alt-MoE: Alt-MoE: Multimodal Alignment via Alternating Optimization of Multi-directional MoE with Unimodal Models](https://www.arxiv.org/abs/2409.05929)
## 2023
0. Arxiv [geometry-aware: Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence](https://arxiv.org/abs/2311.17034) [[Code](https://github.com/Junyi42/geoaware-sc)]
1. Arxiv [Lyrics: Lyrics: Boosting Fine-grained Language-Vision Alignment and Comprehension via Semantic-aware Visual Objects](https://arxiv.org/abs/2312.05278)
## 2022
0. NeurIPS [Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning](https://arxiv.org/abs/2203.02053) (展示了模态表征差异的现象，并证明了这一现象在不同的数据模态和神经网络架构中普遍存在) [[Code](https://github.com/weixin-liang/modality-gap)]



