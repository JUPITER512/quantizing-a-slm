% APPENDIX REFERENCE LIST — the exam asks for a complete list of ALL sources you consulted, mainly academic.
% - Lines starting with % are ignored (like this one). Remove the % to include an entry; add % to drop one.
% - Included by default: datasets, model reports, papers cited on the poster, the statistical methods that
%   scripts/analyze.py implements, and the software used. Background papers are commented out:
%   switch them on only if you actually read them.
% - Open every link once and check authors, year and venue before submitting (several 2026 entries are
%   preprints; mark them as preprints, as below).
% - Build:  .venv\Scripts\python.exe poster\build_appendix.py

# References

## Datasets
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457. https://arxiv.org/abs/1803.05457 (data licence CC BY-SA 4.0)
Yubo Wang, Xueguang Ma, Ge Zhang, et al. 2024a. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In NeurIPS 2024 Datasets and Benchmarks Track. https://arxiv.org/abs/2406.01574 (data licence MIT)

## Models (technical reports)
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. 2024. The Llama 3 Herd of Models. arXiv:2407.21783. https://arxiv.org/abs/2407.21783
Qwen Team: An Yang, Baosong Yang, Beichen Zhang, et al. 2024. Qwen2.5 Technical Report. arXiv:2412.15115. https://arxiv.org/abs/2412.15115
Marah Abdin et al. 2024. Phi-4 Technical Report. arXiv:2412.08905. https://arxiv.org/abs/2412.08905
Microsoft: Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, et al. 2025. Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs. arXiv:2503.01743. https://arxiv.org/abs/2503.01743

## Answer flipping, sycophancy and conformity
Philippe Laban, Lidiya Murakhovs'ka, Caiming Xiong, and Chien-Sheng Wu. 2024. Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment. arXiv:2311.08596 (v2, February 2024; v1 November 2023). https://arxiv.org/abs/2311.08596
Mrinank Sharma, Meg Tong, Tomasz Korbak, et al. 2024. Towards Understanding Sycophancy in Language Models. In ICLR 2024. https://arxiv.org/abs/2310.13548
Yibo Hu and Jiaming Qu. 2026. Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks. Preprint, arXiv:2607.05545. https://arxiv.org/abs/2607.05545
% (removed 29.09.2026: not cited or used) Aaron Fanous, Jacob Goldberg, Ank A. Agarwal, Joanna Lin, Anson Zhou, Roxana Daneshjou, and Sanmi Koyejo. 2025. SycEval: Evaluating LLM Sycophancy. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, 8(1):893–900. https://doi.org/10.1609/aies.v8i1.36598
% Jiseung Hong et al. 2025. Measuring Sycophancy of Language Models in Multi-turn Dialogues (SYCON-Bench). arXiv:2505.23840. https://arxiv.org/abs/2505.23840
% Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V. Le. 2023. Simple Synthetic Data Reduces Sycophancy in Large Language Models. arXiv:2308.03958. https://arxiv.org/abs/2308.03958
% Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, et al. 2022. Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251. https://arxiv.org/abs/2212.09251
% No Usable Linear "Capitulation Direction" in Two Small LLMs: A Validation Protocol for Activation-Steering Claims, and a Cross-Family Behavioral Study of Sycophancy Under Pushback. 2026. Preprint, arXiv:2609.17550. https://arxiv.org/abs/2609.17550

## Quantization
Junyuan Hong et al. 2024. Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression. In ICML 2024. https://arxiv.org/abs/2403.15447
Irina Proskurina, Luc Brun, Guillaume Metzler, and Julien Velcin. 2024. When Quantization Affects Confidence of Large Language Models? In Findings of NAACL 2024, pages 1918–1928. https://arxiv.org/abs/2405.00632
Yao Fu, Xianxuan Long, Runchao Li, et al. 2025. Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of Quantized LLMs. arXiv:2508.19432. https://arxiv.org/abs/2508.19432
% Zhichao Xu, Ashim Gupta, Tao Li, Oliver Bentham, and Vivek Srikumar. 2024. Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression. In Findings of EMNLP 2024, pages 15359–15396. https://aclanthology.org/2024.findings-emnlp.901/
% Federico Marcuzzi et al. 2026. How Quantization Shapes Bias in Large Language Models. In Proceedings of EACL 2026. https://aclanthology.org/2026.eacl-long.17/
% P. K. Rath and R. Maliakkal. 2026. Quantization Undoes Alignment: Bias Emergence in Compressed LLMs Across Models and Precision Levels. Preprint, arXiv:2605.15208. https://arxiv.org/abs/2605.15208
% Y. Fu et al. 2026. When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs. Preprint, arXiv:2608.25977. https://arxiv.org/abs/2608.25977
% Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct. 2026. Preprint, arXiv:2601.14277. https://arxiv.org/abs/2601.14277

## Measurement validity
Xinpeng Wang, Bolei Ma, Chengzhi Hu, Leon Weber-Genzel, Paul Röttger, Frauke Kreuter, Dirk Hovy, and Barbara Plank. 2024b. "My Answer is C": First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models. In Findings of ACL 2024, pages 7407–7416. https://aclanthology.org/2024.findings-acl.441/
% Xinpeng Wang, Chengzhi Hu, Bolei Ma, Paul Röttger, and Barbara Plank. 2024. Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You Think. arXiv:2404.08382. https://arxiv.org/abs/2404.08382
% Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Rose Kirk, Hinrich Schütze, and Dirk Hovy. 2024. Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models. In Proceedings of ACL 2024, pages 15295–15311. https://aclanthology.org/2024.acl-long.816/
% Simon Münker. 2025. Fingerprinting LLMs through Survey Item Factor Correlation: A Case Study on Humor Style Questionnaire. In Proceedings of EMNLP 2025, pages 245–258. https://aclanthology.org/2025.emnlp-main.13/
% N. Schwager, C. Hau, S. Münker, and A. Rettinger. 2026. The Unsampled Truth: Psychometrics in SLMs Measure Prompt Artifacts, Not Psychological Constructs. Preprint, arXiv:2606.03357. https://arxiv.org/abs/2606.03357
% Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? In FAccT '21, pages 610–623. https://doi.org/10.1145/3442188.3445922

## Statistical methods
Quinn McNemar. 1947. Note on the Sampling Error of the Difference between Correlated Proportions or Percentages. Psychometrika, 12(2):153–157. https://doi.org/10.1007/BF02295996
Sture Holm. 1979. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2):65–70.
Daniël Lakens. 2017. Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses. Social Psychological and Personality Science, 8(4):355–362. https://doi.org/10.1177/1948550617697177
Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to the Bootstrap. Chapman & Hall/CRC.
Kung-Yee Liang and Scott L. Zeger. 1986. Longitudinal Data Analysis Using Generalized Linear Models. Biometrika, 73(1):13–22. https://doi.org/10.1093/biomet/73.1.13
James A. Hanley and Barbara J. McNeil. 1982. The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve. Radiology, 143(1):29–36. https://doi.org/10.1148/radiology.143.1.7063747
Frank Wilcoxon. 1945. Individual Comparisons by Ranking Methods. Biometrics Bulletin, 1(6):80–83.
Jacob Cohen. 1960. A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement, 20(1):37–46. https://doi.org/10.1177/001316446002000104
% (removed 29.09.2026 with Cohen's h, which is no longer computed)
% Jacob Cohen. 1988. Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum.
% (removed 29.09.2026 with ECE, which is no longer computed)
% Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On Calibration of Modern Neural Networks. In ICML 2017. https://arxiv.org/abs/1706.04599
Edwin B. Wilson. 1927. Probable Inference, the Law of Succession, and Statistical Inference. Journal of the American Statistical Association, 22(158):209–212. https://doi.org/10.1080/01621459.1927.10502953
% Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht. 2015. Obtaining Well Calibrated Probabilities Using Bayesian Binning. In AAAI 2015.

## Software
Ollama. https://github.com/ollama/ollama (API documentation: https://docs.ollama.com/api/chat)
Georgi Gerganov et al. llama.cpp and the GGUF format. https://github.com/ggml-org/llama.cpp
Skipper Seabold and Josef Perktold. 2010. statsmodels: Econometric and Statistical Modeling with Python. In Proceedings of the 9th Python in Science Conference, pages 92–96.
Fabian Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12:2825–2830.
Pauli Virtanen et al. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272.
Wes McKinney. 2010. Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference, pages 56–61.
John D. Hunter. 2007. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3):90–95.
Quentin Lhoest et al. 2021. Datasets: A Community Library for Natural Language Processing. In EMNLP 2021 System Demonstrations.
