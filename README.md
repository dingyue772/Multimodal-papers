This is a repo for organizing papers related to MLLMs. Most papers are from AK's daily papers, and I'll writing reading notes from time to time.

# Daily Papers

1. Arxiv [MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning](https://arxiv.org/abs/2409.20566) (使用MM1的模型架构，采用以数据为中心的模型训练方法，系统探索了整个模型训练生命周期中不同数据混合的影响，推出MoE-based MLLMs, 处理视频理解的MM1.5-Video和处理移动UI理解的MM1.5-UI)
2. NeurIPS [Image Copy Detection for Diffusion Models](https://arxiv.org/abs/2409.19952) (引入了首个专为扩散模型设计的图像复制检测（ICD）模型ICDiff并使用Stable Diffusion v1.5创建了D-Rep数据集) [[Code](https://github.com/WangWenhao0716/PDF-Embedding)]
3. Arxiv [IDEAW: Robust Neural Audio Watermarking with Invertible Dual-Embedding](https://arxiv.org/abs/2409.19627) (为解决当前音频水印技术中存在的容量低、隐蔽性不够和水印定位问题，设计了一个双嵌入水印模型) [[Code](https://github.com/PecholaL/IDEAW)]
4. NeurIPS [One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos](https://arxiv.org/abs/2409.19603) (为解决视频中的语言指导推理分割问题，将LLM与SAM结合提出了多模态大语言模型VideoLISA) [[Code](https://github.com/showlab/VideoLISA)]
5. Arxiv [Illustrious: an Open Advanced Illustration Model](https://arxiv.org/abs/2409.19946) (提出了动漫图像生成模型Illustrious，深入探讨了三种模型改进方法：首先是batch size和 dropout，然后是训练图像分辨率以及训练数据中的多粒度文本) (**Notes**：训练数据中的多粒度文本咋怎么用的)
6. Arxiv [ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer](https://arxiv.org/abs/2410.00086) (https://arxiv.org/abs/2410.00086) (研究多模态条件引导的视觉生成模型，通过统一生成模型的条件格式并引入一个新的基于Transformer 的扩散模型，提出了ACE(All-round Creator and Editor)模型) [[Code](https://ali-vilab.github.io/ace-page/)]
7. Arxiv [Visual Context Window Extension: A New Perspective for Long Video Understanding](https://arxiv.org/abs/2409.20018) (https://arxiv.org/abs/2409.20018) (从上下文窗口的角度来解决长视频理解的挑战，提出通过扩展视觉上下文窗口来适应长视频理解任务，不需要在大规模长视频数据集上重新训练模型，还引入了引入了一种渐进式池化推理策略以减少视觉标记的数量) [[Code](https://hcwei13.github.io/Visual-Context-Window-Extension/)] **[[ReadNotes]]** (**Notes**: 视觉标记和文本标记的上下文窗口有什么不同，能够定量分析，这样的不同会带来什么影响)
8. EMNLP [Visual Question Decomposition on Multimodal Large Language Models](https://arxiv.org/abs/2409.19339) (探索多模态大规模语言模型（MLLMs）的问题分解能力，引入一个系统评估框架，包括一个数据集和若干评估标准，一个微调数据集DecoVQA+，以及一个高效的微调流程) [[Code](https://github.com/freesky01/Visual-Question-Decomposition)] [[ReadNotes](https://zhuanlan.zhihu.com/p/814152843)]
9. EMNLP Findings [LongGenBench: Long-context Generation Benchmark](https://arxiv.org/abs/2410.04199) (探索大语言模型长文本生成能力的合成数据集) [[Code](https://github.com/Dominic789654/LongGenBench)] [[**ReadNotes**]]
10. arXiv [Only-IF:Revealing the Decisive Effect of Instruction Diversity on Generalization](https://arxiv.org/abs/2410.04717) (探索大语言模型的任务泛化能力与微调时数据指令多样性之间的关系) [[**ReadNotes**]] (Notes: 大语言模型的泛化能力是如何评测的，衡量指令多样性的指标)
11. arXiv [RevisEval: Improving LLM-as-a-Judge via Response-Adapted References](https://arxiv.org/abs/2410.05193) (将评判后的response作为LLM-judge接下来评判的参考，通过这样的方式来改进单纯使用LLM作为自然语言生成任务judge的评估范式)
12. arXiv [MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions](https://arxiv.org/abs/2410.02743) (一种新的RLHF范式，克服传统token-level RLHF的缺陷，比如长序列中的奖励指定，这会严重影响训练的efficiency，提出了这样一种结合更高粒度action的RLHF，提升模型训练效率和训练后模型的性能)[[Code](https://github.com/ernie-research/MA-RLHF)]

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



