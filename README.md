# Token-Efficient-Leverage-Learning

Welcome to the official GitHub repository for the paper "Token-Efficient Leverage Learning in Large Language Models" available at [https://arxiv.org/abs/2404.00914](https://arxiv.org/abs/2404.00914). This repository hosts datasets and essential tools involved in the study, including a modified version of the Icelandic-English translation dataset from WMT-21 and the Halu-CoT-JSON dataset adapted from [https://github.com/RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval). Detailed modification methods are thoroughly discussed in the paper.

In our paper, "Token-Efficient Leverage Learning in Large Language Models", we introduce and detail a minimalistic implementation of leverage learning, named Token-Efficient Leverage Learning (TELL). 

The core idea of leverage learning is that a task often consists of task-specific and non-specific capabilities. Traditionally, massive datasets are used to fine-tune both these capabilities, which could lead to inefficient use of valuable training data. Leverage learning suggests that it is possible to strategically use minimal task-specific data to enhance task-specific capabilities, while non-specific capabilities can be learned from more general data.

While the concept is straightforward, its implementation faces challenges. Firstly, large language models (LLMs) do not differentiate between task-specific and general data, nor do they recognize which capabilities are task-specific. This could lead to a degradation back to conventional Supervised Fine-Tuning (SFT). 

Our implementation of TELL addresses these issues by appending an anchor prompt to task data to inject consistent semantic features, tackling the first challenge. We leverage the quantization hypothesis to demonstrate that when general data is abundant and randomly mixed, the optimization of task data for task-specific capabilities can approach an optimal sequence.

In low-resource settings, TELL can activate tasks that are unfeasible with conventional methods; for tasks that are feasible, TELL achieves significantly better outcomes with the same amount of data. If performance parity is the goal, TELL drastically reduces the data required compared to traditional methods, often by nearly an order of magnitude, achieving similar or superior results. A notable example is that just 10 domain-specific data entries (10^4 tokens) can substantially enhance LLM performance in that domain.

It's important to note that the traditional methods referenced here generally include LoRA, which is a top-tier baseline in Parameter-Efficient Fine-Tuning (PEFT) and has been proven to outperform full-parameter fine-tuning in low-resource settings. This work potentially makes various applications of fine-tuning in low-resource scenarios (such as user-customized models) more feasible.
