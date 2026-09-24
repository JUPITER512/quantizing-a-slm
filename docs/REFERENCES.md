# References and Previous Work

Project: *Cave In at 4 Bits? Quantization Precision, Speaker-Free Pressure, and First-Token Confidence in Answer Flipping of Local LLMs.*

Every entry has a full citation (ACL style), its links, and what it is used for. Entries marked **(preprint)** are not peer-reviewed; several 2026 preprints were read only as abstracts during the search, so open each link once before citing it. The appendix must list only sources you actually consulted, mainly academic.

Datasets and models have their own files: [`DATASETS.md`](DATASETS.md) and [`MODELS.md`](MODELS.md).

---

## 1. Seminar papers this project connects to

| Citation | Links | Used for |
|---|---|---|
| Xinpeng Wang, Bolei Ma, Chengzhi Hu, Leon Weber-Genzel, Paul Röttger, Frauke Kreuter, Dirk Hovy, and Barbara Plank. 2024. "My Answer is C": First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models. In *Findings of ACL 2024*, pages 7407–7416. | [ACL Anthology](https://aclanthology.org/2024.findings-acl.441/) · [arXiv:2402.14499](https://arxiv.org/abs/2402.14499) · [code](https://github.com/mainlp/MCQ-Mismatch) | Validity check: first-token letter vs text letter |
| Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Rose Kirk, Hinrich Schütze, and Dirk Hovy. 2024. Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models. In *Proceedings of ACL 2024*, pages 15295–15311. | [ACL Anthology](https://aclanthology.org/2024.acl-long.816/) · [arXiv:2402.16786](https://arxiv.org/abs/2402.16786) | Forced-choice validity; motivates the paraphrase control |
| Simon Münker. 2025. Fingerprinting LLMs through Survey Item Factor Correlation: A Case Study on Humor Style Questionnaire. In *Proceedings of EMNLP 2025*, pages 245–258. | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.13/) | Measurement validity of LLM behaviour |
| Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? In *FAccT '21*, pages 610–623. | [DOI](https://doi.org/10.1145/3442188.3445922) | Deployment-harm framing |

---

## 2. Pushback, sycophancy and conformity

| Citation | Links | What it found / used for |
|---|---|---|
| Philippe Laban, Lidiya Murakhovs'ka, Caiming Xiong, and Chien-Sheng Wu. 2024. Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment. | [arXiv:2311.08596](https://arxiv.org/abs/2311.08596) | Ten LLMs flip 46% on average after "Are you sure?"; 17% accuracy drop. Origin of the `reask` condition |
| Mrinank Sharma, Meg Tong, Tomasz Korbak, et al. 2024. Towards Understanding Sycophancy in Language Models. In *ICLR 2024*. | [arXiv:2310.13548](https://arxiv.org/abs/2310.13548) | Assistants swayed by user pushback, not only when unsure |
| Aaron Fanous, Jacob Goldberg, Ank A. Agarwal, Joanna Lin, Anson Zhou, Roxana Daneshjou, and Sanmi Koyejo. 2025. SycEval: Evaluating LLM Sycophancy. In *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, 8(1):893–900. | [DOI](https://doi.org/10.1609/aies.v8i1.36598) · [arXiv:2502.08177](https://arxiv.org/abs/2502.08177) | Progressive vs regressive sycophancy; authority/citation rebuttals strongest. Origin of the `expert` condition |
| Jiseung Hong et al.. 2025. Measuring Sycophancy of Language Models in Multi-turn Dialogues (SYCON-Bench). | [arXiv:2505.23840](https://arxiv.org/abs/2505.23840) | Multi-turn metrics (Turn-of-Flip, Number-of-Flips); possible extension |
| Yibo Hu and Jiaming Qu. 2026. Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks. **(preprint)** | [arXiv:2607.05545](https://arxiv.org/abs/2607.05545) · [code](https://github.com/yibo-hu-lab/llm-speaker-free-floor) | Speaker-free assertion causes 66.5% harmful revision vs 10.3% plain re-ask. Backbone of RQ2 and the `speaker_free` condition |
| No Usable Linear "Capitulation Direction" in Two Small LLMs: A Validation Protocol for Activation-Steering Claims, and a Cross-Family Behavioral Study of Sycophancy Under Pushback. 2026. **(preprint)** | [arXiv:2609.17550](https://arxiv.org/abs/2609.17550) | Qwen2.5-1.5B and Llama-3.2-1B flip 41.8% / 43.1% on TriviaQA; no precision variation. Shows small-model pushback is studied — precision is what is new |
| Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V. Le. 2023. Simple Synthetic Data Reduces Sycophancy in Large Language Models. | [arXiv:2308.03958](https://arxiv.org/abs/2308.03958) · [code](https://github.com/google/sycophancy-intervention) | Background: where sycophancy comes from |
| Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, et al. 2022. Discovering Language Model Behaviors with Model-Written Evaluations. | [arXiv:2212.09251](https://arxiv.org/abs/2212.09251) | Background: early sycophancy evaluations |

---

## 3. Quantization and trust, confidence, bias, personality

| Citation | Links | What it found / used for |
|---|---|---|
| **Y. Fu et al. 2025. Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of Quantized LLMs.** | [arXiv:2508.19432](https://arxiv.org/abs/2508.19432) | Quantized models susceptible to misleading *system* prompts; names user-side deception in quantized LLMs as future work. **Closest paper — cite it in the gap statement** |
| Junyuan Hong et al. 2024. Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression. In *ICML 2024*. | [arXiv:2403.15447](https://arxiv.org/abs/2403.15447) | 4-bit keeps the original's trustworthiness; 3-bit degrades. Grounds the "small effect" expectation |
| Irina Proskurina, Luc Brun, Guillaume Metzler, and Julien Velcin. 2024. When Quantization Affects Confidence of Large Language Models? In *Findings of NAACL 2024*, pages 1918–1928. | [arXiv:2405.00632](https://arxiv.org/abs/2405.00632) | 4-bit lowers confidence in true labels, mostly where the full model was unsure. Basis of H3b |
| Zhichao Xu, Ashim Gupta, Tao Li, Oliver Bentham, and Vivek Srikumar. 2024. Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression. In *Findings of EMNLP 2024*, pages 15359–15396. | [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.901/) · [arXiv:2407.04965](https://arxiv.org/abs/2407.04965) | Compression effects beyond perplexity |
| Federico Marcuzzi et al. 2026. How Quantization Shapes Bias in Large Language Models. In *Proceedings of EACL 2026*. | [ACL Anthology](https://aclanthology.org/2026.eacl-long.17/) · [arXiv:2508.18088](https://arxiv.org/abs/2508.18088) | Bias effects depend on metric type (probability vs text) |
| P. K. Rath and R. Maliakkal. 2026. Quantization Undoes Alignment: Bias Emergence in Compressed LLMs Across Models and Precision Levels. **(preprint)** | [arXiv:2605.15208](https://arxiv.org/abs/2605.15208) | 2.5–5.6% of BBQ items develop new bias at 4-bit although perplexity barely changes: item-level shifts hide behind stable aggregates |
| Y. Fu et al. 2026. When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs. **(preprint)** | [arXiv:2608.25977](https://arxiv.org/abs/2608.25977) | Personality profiles across quantization methods |
| QuantiBias: Benchmarking Quantization-Induced Bias in LLMs. 2026. **(preprint)** | [arXiv:2607.21063](https://arxiv.org/abs/2607.21063) | Bias benchmark under quantization; shows that topic is taken |
| Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct. 2026. **(preprint)** | [arXiv:2601.14277](https://arxiv.org/abs/2601.14277) | Quality of GGUF k-quants (the formats you use) |

---

## 4. Measurement validity

| Citation | Links | Used for |
|---|---|---|
| N. Schwager, C. Hau, S. Münker, and A. Rettinger. 2026. The Unsampled Truth: Psychometrics in SLMs Measure Prompt Artifacts, Not Psychological Constructs. **(preprint)** | [arXiv:2606.03357](https://arxiv.org/abs/2606.03357) | Small-model outputs dominated by prompt artifacts — motivates the paraphrase and order controls (from your lecturer's group) |
| Xinpeng Wang, Chengzhi Hu, Bolei Ma, Paul Röttger, and Barbara Plank. 2024. Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You Think. | [arXiv:2404.08382](https://arxiv.org/abs/2404.08382) · [code](https://github.com/mainlp/MCQ-Robustness) | Text answers vs first-token answers; discussion |
| L. Debevc, N. Chatterjee, et al. 2026. Navigating the Digital Spectrum: Assessing Political Bias, Stability, and Downstream Fairness in LLMs. **(preprint)** | [arXiv:2609.08637](https://arxiv.org/abs/2609.08637) | Political-compass stability across quantization; why that topic was not chosen |
| S. Kamal et al. 2025. A Detailed Factor Analysis for the Political Compass Test. | [arXiv:2506.22493](https://arxiv.org/abs/2506.22493) | Background |

---

## 5. Statistics and analysis methods

| Citation | Link | Method |
|---|---|---|
| Bradley Efron and Robert J. Tibshirani. 1993. *An Introduction to the Bootstrap*. Chapman & Hall/CRC. | — | Item-clustered bootstrap CIs |
| Quinn McNemar. 1947. Note on the Sampling Error of the Difference between Correlated Proportions or Percentages. *Psychometrika*, 12(2):153–157. | [DOI](https://doi.org/10.1007/BF02295996) | Exact McNemar test (primary test) |
| Sture Holm. 1979. A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2):65–70. | — | Holm correction |
| Daniël Lakens. 2017. Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses. *Social Psychological and Personality Science*, 8(4):355–362. | [DOI](https://doi.org/10.1177/1948550617697177) | TOST equivalence (H1b) |
| Kung-Yee Liang and Scott L. Zeger. 1986. Longitudinal Data Analysis Using Generalized Linear Models. *Biometrika*, 73(1):13–22. | [DOI](https://doi.org/10.1093/biomet/73.1.13) | GEE with item clusters |
| Douglas Bates, Martin Mächler, Ben Bolker, and Steve Walker. 2015. Fitting Linear Mixed-Effects Models Using lme4. *Journal of Statistical Software*, 67(1). | [DOI](https://doi.org/10.18637/jss.v067.i01) | Mixed-model robustness check (method reference) |
| James A. Hanley and Barbara J. McNeil. 1982. The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve. *Radiology*, 143(1):29–36. | [DOI](https://doi.org/10.1148/radiology.143.1.7063747) | AUROC (H3a) |
| Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On Calibration of Modern Neural Networks. In *ICML 2017*. | [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | Expected Calibration Error |
| Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht. 2015. Obtaining Well Calibrated Probabilities Using Bayesian Binning. In *AAAI 2015*. | — | ECE binning |
| Frank Wilcoxon. 1945. Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6):80–83. | — | Paired confidence difference (H3b) |
| Jacob Cohen. 1988. *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum. | — | Cohen's h |
| Jacob Cohen. 1960. A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1):37–46. | [DOI](https://doi.org/10.1177/001316446002000104) | Cohen's κ (validity check) |

---

## 6. Software

| Citation | Link |
|---|---|
| Ollama | [github.com/ollama/ollama](https://github.com/ollama/ollama) · [API reference](https://docs.ollama.com/api/chat) · [top-k logprobs issue](https://github.com/ollama/ollama/issues/18579) |
| llama.cpp and the GGUF format | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Skipper Seabold and Josef Perktold. 2010. statsmodels: Econometric and Statistical Modeling with Python. In *Proceedings of the 9th Python in Science Conference*, pages 92–96. | [statsmodels.org](https://www.statsmodels.org/) |
| Fabian Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. *JMLR*, 12:2825–2830. | [scikit-learn.org](https://scikit-learn.org/) |
| Pauli Virtanen et al. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17:261–272. | [scipy.org](https://scipy.org/) |
| Wes McKinney. 2010. Data Structures for Statistical Computing in Python. In *Proceedings of the 9th Python in Science Conference*, pages 56–61. | [pandas.pydata.org](https://pandas.pydata.org/) |
| John D. Hunter. 2007. Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3):90–95. | [matplotlib.org](https://matplotlib.org/) |
| Quentin Lhoest et al. 2021. Datasets: A Community Library for Natural Language Processing. In *EMNLP 2021 System Demonstrations*. | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets) |

---

## 7. Which references go on the poster

Poster footer (short form): Laban et al. 2024 · Hu & Qu 2026 · Fu et al. 2025 · Proskurina et al. 2024 · Hong et al. 2024 · Wang et al. 2024. Everything else goes only into the appendix, and only if you used it.
