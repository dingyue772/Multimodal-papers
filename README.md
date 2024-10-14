This is a repo for organizing papers related to Multimodal Learning. Most papers are from AK's daily papers, and I'll writing reading notes from time to time.

# Diffusion Model

1. arXiv [DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation](https://arxiv.org/abs/2410.08159) (提出DART文生图框架，是一种结合自回归和扩散方法的非马尔可夫框架，不依赖于图像量化的能够处理文本和图像的统一框架) 
2. arXiv [Progressive Autoregressive Video Diffusion Models](https://arxiv.org/abs/2410.08151) (解决视频扩散模型生成视频时长短的问题，通过引入渐进噪声分配和渐进视频去噪，提出了自回归式的视频扩散模型) [[Code](https://desaixie.github.io/pa-vdm/)]
3. arXiv [Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow](https://arxiv.org/abs/2410.07303) (简化Retified flow的核心组件1 2，即流匹配的扩散形式和v-prediction，关键训练目标为一节近似ODE路径。提出了Rectified Diffustion，简化基于校正流的工作流程并实现了更优的性能) [[Code](https://github.com/G-U-N/Rectified-Diffusion)]
4. arXiv [DICE: Discrete Inversion Enabling Controllable Editing for Multinomial Diffusion and Masked Generative Models](https://arxiv.org/abs/2410.08207) (DICE, 首个条件内容编辑的离散扩散模型) [[Code](https://hexiaoxiao-cs.github.io/DICE/)]
5. Arxiv [ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer](https://arxiv.org/abs/2410.00086) (https://arxiv.org/abs/2410.00086) (研究多模态条件引导的视觉生成模型，通过统一生成模型的条件格式并引入一个新的基于Transformer 的扩散模型，提出了ACE(All-round Creator and Editor)模型) [[Code](https://ali-vilab.github.io/ace-page/)]
6. Arxiv [Illustrious: an Open Advanced Illustration Model](https://arxiv.org/abs/2409.19946) (提出了动漫图像生成模型Illustrious，深入探讨了三种模型改进方法：首先是batch size和 dropout，然后是训练图像分辨率以及训练数据中的多粒度文本) (**Notes**：训练数据中的多粒度文本咋怎么用的)
7. NeurIPS [Image Copy Detection for Diffusion Models](https://arxiv.org/abs/2409.19952) (引入了首个专为扩散模型设计的图像复制检测（ICD）模型ICDiff并使用Stable Diffusion v1.5创建了D-Rep数据集) [[Code](https://github.com/WangWenhao0716/PDF-Embedding)]

# LLMs

1. arXiv [MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code](https://arxiv.org/abs/2410.08196) (构建了一个新的直接针对数学推理的数据集MathCode-Pile，token数量达19.2B，包含为自然语言推理步骤与其对应代码的配对数据。使用该数据集进行增量预训练，得到MathCoder2系列模型) [[Code](https://mathllm.github.io/mathcoder2/)]  
2. arXiv [PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers in LLMs](https://arxiv.org/abs/2410.05265) (LLM量化，提出PrefixQuant，在KV cache中为高频异常token添加前缀，避免token级别的动态量化，同时实现性能提升和推理加速) [[Code](https://github.com/ChenMnZ/PrefixQuant)]
3. arXiv [Benchmarking Agentic Workflow Generation](https://arxiv.org/abs/2410.07869) (针对LLM的复杂问题分解过程，提出了问题分解工作流生成数据集WORFBENCH和一个系统的评估协议WORFEVAL) [[Code]]
4. arXiv [Towards Self-Improvement of LLMs via MCTS: Leveraging Stepwise Knowledge with Curriculum Preference Learning](https://arxiv.org/abs/2410.06508) (解决当前蒸馏方法并未充分利用MCTS中丰富轨迹信息的问题，提出AlphaLLM-CPL方法，利用轨迹对信息和课程偏好学习策略提升LLM的推理能力)
5. arXiv [Emergent properties with repeated examples](https://arxiv.org/abs/2410.07041) (考察了训练样本重复次数对模型性能的影响，针对数学问题（最大公约数、模乘法和矩阵特征值）进行实验，得到了一些有趣的发现)
6. arXiv [Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates](https://arxiv.org/abs/2410.07137) (揭示LLM评估基准中的作弊可能性和作弊输出的可转移性) [[Code](https://github.com/sail-sg/Cheating-LLM-Benchmarks)]
7. arXiv [Accelerated Preference Optimization for Large Language Model Alignment](https://arxiv.org/abs/2410.06293) (利用动量方法加速RLHF DPO算法。提出加速偏好优化（APO）框架，将迭代偏好优化方法视为近端点方法，并采用Nesterov动量技术加速LLMs的对齐，带来更快的收敛速度)
8. EMNLP [Data Advisor: Dynamic Data Curation for Safety Alignment of Large Language Models](https://arxiv.org/abs/2410.05269) (解决LLM合成数据质量低下的问题，提出一种增强型LLM数据生成方法Data Advisor，考虑所需数据集的特征，通过一系列预定义原则，监控生成数据的状态，识别当前数据集的弱点，并为下一次数据生成提供建议) [[Code](https://feiwang96.github.io/DataAdvisor/)]
9. arXiv [Vector-ICL: In-context Learning with Continuous Vector Representations](https://arxiv.org/abs/2410.05629) (研究探讨是否可以将LLM的上下文学习能力扩展到从黑箱预训练编码器获得的多领域连续向量。通过轻量级投影器将输入数据对齐到LLM的嵌入空间，观察到LLM对这些投影向量的处理) [[Code](https://github.com/EvanZhuang/vector-icl)] **ReadNotes 2**
10. arXiv [Zebra: In-Context and Generative Pretraining for Solving Parametric PDEs](https://arxiv.org/abs/2410.03437) (解决时间依赖参数化偏微分方程（PDEs）问题，引入Zebra模型，通过在输入序列中结合上下文轨迹或前状态进行条件化，从而动态适应新任务。能够灵活处理任意大小的上下文输入，并通过采样多个解轨迹支持不确定性量化。) [[Code](https://github.com/LouisSerrano/zebra)]
11. arXiv [MEXA: Multilingual Evaluation of English-Centric LLMs via Cross-Lingual Alignment](https://arxiv.org/abs/2410.05873) (利用配对多语言数据计算语言对齐程度，评估语言理解从英语到其他语言的转移) [[Code](https://github.com/cisnlp/mexa)] **ReadNote** 
12. arXiv [Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks in Superposition](https://arxiv.org/abs/2410.05603) (探索LLM的“任务叠加”能力，即LLMs可以在单次推理调用中同时执行多个计算上不同的ICL任务) [[Code](https://github.com/edixiong/task-superposition)]
13. arXiv [Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://arxiv.org/abs/2410.08115) (解决基于LLM的MAS在协作问题解决方面遇到的挑战，通过引入Optima框架，提升通讯效率、任务有效性，并探索多种强化学习算法，并结合MTCS技术进一步优化) [[Code](https://chenweize1998.github.io/optima-project-page/)]
14. arXiv [SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe](https://arxiv.org/abs/2410.05248) (改进指令微调阶段的next-token-prediction, NTP loss function， 考虑LLMs在语义表示空间中表现出不均匀的信心程度，应用基于Mixup的正则化以减缓对高信心例子的过拟合，并传播监督信号以改善对低信心例子的学习)
15. EMNLP Findings [LongGenBench: Long-context Generation Benchmark](https://arxiv.org/abs/2410.04199) (探索大语言模型长文本生成能力的合成数据集) [[Code](https://github.com/Dominic789654/LongGenBench)] [[**ReadNotes**]]
16. arXiv [Only-IF:Revealing the Decisive Effect of Instruction Diversity on Generalization](https://arxiv.org/abs/2410.04717) (探索大语言模型的任务泛化能力与微调时数据指令多样性之间的关系) [[**ReadNotes**]] (Notes: 大语言模型的泛化能力是如何评测的，衡量指令多样性的指标)
17. arXiv [RevisEval: Improving LLM-as-a-Judge via Response-Adapted References](https://arxiv.org/abs/2410.05193) (将评判后的response作为LLM-judge接下来评判的参考，通过这样的方式来改进单纯使用LLM作为自然语言生成任务judge的评估范式)
18. arXiv [MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions](https://arxiv.org/abs/2410.02743) (一种新的RLHF范式，克服传统token-level RLHF的缺陷，比如长序列中的奖励指定，这会严重影响训练的efficiency，提出了这样一种结合更高粒度action的RLHF，提升模型训练效率和训练后模型的性能)[[Code](https://github.com/ernie-research/MA-RLHF)]

# LVLMs

1. Arxiv [MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning](https://arxiv.org/abs/2409.20566) (使用MM1的模型架构，采用以数据为中心的模型训练方法，系统探索了整个模型训练生命周期中不同数据混合的影响，推出MoE-based MLLMs, 处理视频理解的MM1.5-Video和处理移动UI理解的MM1.5-UI)
2. NeurIPS [One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos](https://arxiv.org/abs/2409.19603) (为解决视频中的语言指导推理分割问题，将LLM与SAM结合提出了多模态大语言模型VideoLISA) [[Code](https://github.com/showlab/VideoLISA)]
3. Arxiv [Visual Context Window Extension: A New Perspective for Long Video Understanding](https://arxiv.org/abs/2409.20018) (https://arxiv.org/abs/2409.20018) (从上下文窗口的角度来解决长视频理解的挑战，提出通过扩展视觉上下文窗口来适应长视频理解任务，不需要在大规模长视频数据集上重新训练模型，还引入了引入了一种渐进式池化推理策略以减少视觉标记的数量) [[Code](https://hcwei13.github.io/Visual-Context-Window-Extension/)] **[[ReadNotes]]** (**Notes**: 视觉标记和文本标记的上下文窗口有什么不同，能够定量分析，这样的不同会带来什么影响)
4. EMNLP [Visual Question Decomposition on Multimodal Large Language Models](https://arxiv.org/abs/2409.19339) (探索多模态大规模语言模型（MLLMs）的问题分解能力，引入一个系统评估框架，包括一个数据集和若干评估标准，一个微调数据集DecoVQA+，以及一个高效的微调流程) [[Code](https://github.com/freesky01/Visual-Question-Decomposition)] [[ReadNotes](https://zhuanlan.zhihu.com/p/814152843)]
5. arXiv [Intriguing Properties of Large Language and Vision Models](https://arxiv.org/abs/2410.04751) (针对性地评估LLaVA模型在感知和高级推理任务上的表现，作者发现了一些有趣的现象，例如全局图像处理、数学问题解决能力、跨模态对齐的过拟合以及底层表示空间的重要性) [[Code](https://github.com/passing2961/IP-LLVM)] **ReadNotes 1**
6. EMNLP [Preserving Multi-Modal Capabilities of Pre-trained VLMs for Improving Vision-Linguistic Compositionality]()https://arxiv.org/abs/2410.05210  (针对LVLM中组合理解能力增强而多模态能力下降的问题，提出FSC-CLIP方法，通过整合局部硬负样本损失和选择性校准正则化进行解决) [[Code](https://github.com/ytaek-oh/fsc-clip)] **ReadNotes 4**
7. arXiv [GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models](https://arxiv.org/abs/2410.06154) (提出GLOV这种元提示机制，使用LLM作为VLM的隐式优化器，理解下游任务描述，为VLM生成合适的上下文提示)  **ReadNotes 5**
8. arXiv [WALL-E: World Alignment by Rule Learning Improves World Model-based LLM Agents](https://arxiv.org/abs/2410.07484) (为了让LLM成为一个强大的世界模型，需要解决LLM先验知识和环境动态之间的差距，提出一种世界对齐方法并通过在LLM上进行规则学习高效实现) [[Code](https://github.com/elated-sawyer/WALL-E)] **ReadNotes 3**
9. arXiv [Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate](https://arxiv.org/abs/2410.07167) (大型视觉语言模型（LVLMs）的多模态预训练质量评估问题，提出MIR指标，从模态间分布距离的角度评估预训练质量) [[Code](https://github.com/shikiw/modality-integration-rate)] **ReadNote 0** (Ps: 1. FID这个值本身并没有什么意义，因为他的计算过程强依赖于pretrain data 2. 这个metic用FID感觉还是有点粗糙)
10. arXiv [Aria: An Open Multimodal Native Mixture-of-Experts Model](https://arxiv.org/abs/2410.05993) (引入Aria这个多模态原生模型，专家混合模型，使用四阶段预训练流程，模型具有强大的语言理解、多模态理解、长上下文窗口和指令跟随能力) [[Code](https://github.com/rhymes-ai/Aria/)] [[Website](https://rhymes.ai/)]
11. arXiv [Personalized Visual Instruction Tuning](https://arxiv.org/abs/2410.07113) (提出个性化视觉指令微调（PVIT），自动生成个性化对话数据的复杂管道，用语评估MLLM个性化潜力的基准 P-Bench) [[Code](https://github.com/sterzhang/PVIT)]
12. arXiv [Pixtral 12B](https://arxiv.org/abs/2410.07073) (Mistral家的全新开源LVLM，使用自训练的视觉编码器，以自然分辨率和长宽比处理图像，能够在128K token的长上下文窗口中处理任意数量的图像) [[Code](https://github.com/mistralai/mistral-inference)]
13. arXiv [MM-Ego: Towards Building Egocentric Multimodal LLMs](https://arxiv.org/abs/2410.07177) (探索构建用于第一视角视频理解的多模态基础模型，构造了一个最大的视频上第一视角问答数据集，提供了一个包含629个视频和7,026个问题的第一视角问答基准，一种包含新颖“记忆指针提示”机制的多模态架构)
14. arXiv [Temporal Reasoning Transfer from Text to Video](https://arxiv.org/abs/2410.06166) (VideoLLMs在追踪时间变化和推理时间关系方面的挑战源于LLM本身对时间概念的理解困难，提出“文本时间推理转移”（T3），在不使用任何视频数据的情况下，提高了LongVA-7B的时间理解能力) [[Project](https://video-t3.github.io/)]
15. arXiv [ING-VP: MLLMs cannot Play Easy Vision-based Games Yet](https://arxiv.org/abs/2410.06555) (提出第一个互动游戏为基础的视觉规划基准ING-VP，专为评估MLLMs的空间想象和多步推理能力而设计) [[Code](https://github.com/thisisus7/ing-vp)]
16. arXiv [Multimodal Situational Safety](https://arxiv.org/abs/2410.06172) (开发了多模态情境安全基准（MSSBench），用于评估当前MLLMs的情境安全性能) [[Code](https://github.com/eric-ai-lab/MSSBench)] [[Project](https://mssbench.github.io/)]
17. arXiv [TRACE: Temporal Grounding Video LLM via Causal Event Modeling](https://arxiv.org/abs/2410.05643) (针对视频时间定位问题，引入因果事件建模框架，将视频表示为事件序列，通过前序事件、视频输入和文本指令预测当前事件，提出一种新颖的任务交错视频LLM模型TRACE) [[Code](https://github.com/gyxxyg/trace)]
18. arXiv [Data Selection via Optimal Control for Language Models](https://arxiv.org/abs/2410.07064) (处理大语言模型预训练数据选择问题，将数据选择问题视为广义的最优控制问题，通过庞特里亚金最大值原理（PMP）求解，引入PMP-based数据选择（PDS）框架) [[Code](https://github.com/microsoft/LMOps/tree/main/data_selection)]
19. arXiv [VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks](https://arxiv.org/abs/2410.05160) (探索能够处理多种下游任务的通用嵌入模型，提出一个大规模多模态嵌入基准MMEB，提出VLM2Vec（Vision-Language Model -> Vector）对比训练框架，通过在MMEB上训练，将任何先进的视觉语言模型转换为嵌入模型) [[Code](https://github.com/TIGER-AI-Lab/VLM2Vec)] [[Project](https://tiger-ai-lab.github.io/VLM2Vec/)] **ReadNote** 
20. ACCV [TinyEmo: Scaling down Emotional Reasoning via Metric Projection](https://arxiv.org/abs/2410.07062) (一组专注于情感推理和分类的多模态模型，引入Metric Projector，将将分类任务从语言模型中分离出来，提高训练和推理的效率。提供了一个偏差检测的半自动框架) [[Code](https://github.com/ggcr/TinyEmo)]
21. arXiv [VHELM: A Holistic Evaluation of Vision Language Models](https://arxiv.org/abs/2410.07112) (提出了一个VLMs的整体评估框架VHELM，整合多个数据集，涵盖视觉感知、知识、推理、偏见、公平性、多语言性、鲁棒性、毒性和安全性等九个方面) [[Code](https://github.com/stanford-crfm/helm)]
22. arXiv [Does Spatial Cognition Emerge in Frontier Models?](https://arxiv.org/abs/2410.06468) (提出系统评估模型空间认知的基准SPACE) 
23. arXiv [MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents](https://arxiv.org/abs/2410.03450) (在具身智能检索多模态相关轨迹数据任务中使用MLLM作为检索器以充分考虑轨迹有效性，即MLLM As ReTriever, MART) [[Code]] 
24. Arxiv [Law of Vision Representation in MLLMs](https://arxiv.org/abs/2408.16357) (分析跨模态对齐和视觉表征的视觉对应关系对LVLMs性能的影响) [[Code](https://github.com/bronyayang/law_of_vision_representation_in_mllms)]
25. Arxiv [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) (不同神经网络表示数据的方式正变得越来越统一并进行了相关的分析)
    [[Code](https://github.com/minyoungg/platonic-rep)]
26. Arxiv [SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration in MLLMs](https://arxiv.org/abs/2408.11813) (遵循LLaVA的预训练-微调范式，但在预训练阶段增加另一个对比损失$L_a$，与下一个令牌预测损失$L_g$一起，得到新的优化损失$L=L_g+\lambda L_a$)
27. ICRL [VLAP: Bridging Vision and Language Spaces with Assignment Prediction(Apr. 15, 2024](https://arxiv.org/abs/2404.09632) [[Code](https://github.com/park-jungin/vlap)]
28. Arxiv [V2T Tokenizer: Beyond Text: Frozen Large Language Models in Visual Signal Comprehension](https://arxiv.org/abs/2403.07874) [[Code](https://github.com/zh460045050/v2l-tokenizer)]
29. Arxiv [SoM prompting: List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs](https://arxiv.org/abs/2404.16375) [[Code](https://github.com/zzxslp/som-llava)]
30. Arxiv [AlignGPT: AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability](https://arxiv.org/abs/2405.14129) [[Code](https://github.com/AlignGPT-VL/AlignGPT)]
31. [Visual Prompting: Rethinking Visual Prompting for Multimodal Large Language Models with External Knowledge](https://arxiv.org/abs/2407.04681)
32. Arxiv [ X-VILA: Cross-Modality Alignment for Large Language Model](https://arxiv.org/abs/2405.19335)
33. Arxiv [LexVLA: Unified Lexical Representation for Interpretable Visual-Language Alignment](https://arxiv.org/abs/2407.17827)
34. ICLR [ Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
35. Arxiv [SIMA: Enhancing Visual-Language Modality Alignment in Large Vision Language Models via Self-Improvement](https://arxiv.org/abs/2405.15973) [[Code](https://github.com/umd-huang-lab/sima)]
36. ICML [WCA: Visual-Text Cross Alignment: Refining the Similarity Score in Vision-Language Models(Jun 5, 2024)](https://arxiv.org/abs/2406.02915) [[Code](https://github.com/tmlr-group/wca)]
37. Arxiv [Multi-Modal Adapter: Multi-Modal Adapter for Vision-Language Models](https://www.arxiv.org/abs/2409.02958)
38. Arxiv [Alt-MoE: Alt-MoE: Multimodal Alignment via Alternating Optimization of Multi-directional MoE with Unimodal Models](https://www.arxiv.org/abs/2409.05929)

39. NeurIPS [Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning](https://arxiv.org/abs/2203.02053) (展示了模态表征差异的现象，并证明了这一现象在不同的数据模态和神经网络架构中普遍存在) [[Code](https://github.com/weixin-liang/modality-gap)]



