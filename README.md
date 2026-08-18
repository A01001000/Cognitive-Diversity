# Cognitive Reasoning Diversity for Robust Juries

## TL;DR
* Researchers have suggested that Human-AI juries may be more robust to judge hacking due to the complementarity of their orthogonal, uncorrelated blind spots (Voudouris, et al., 2026; Jain et al., 2025).
* Hypothesis: Simulating these juries in silico by having diverse cognitive reasoning strategies represented amongst judges may be complementary and thus mitigate judge hacking. 
* A pattern-matching judge is vulnerable to semantic traps that exploit its attention to heuristics, while a causal reasoning judge that does not explicitly mentalize is vulnerable to referential traps that exploit its inability to separate its own knowledge from the more limited mental states of other agents.
* With a 10% lower error rate, juries that vary in terms of cognitive reasoning strategy seem to be more robust than those that simply vary in terms of model architecture and provider.
* Probing and prompting LLMs to reason in a specific way are insufficient methods to induce true cognitive orthogonality, resulting in model functionality leakage.
* Asymmetric Narrow Fine-Tune training with LoRA that uses task-steering prefixes and targets the model's MLP layers yields an over 4% accuracy gain for a cognitively diverse jury over individual Pattern and Causal Judge models, indicating that orthogonality can be learned. This further supports the complementarity of orthogonal blind spots in cognitive reasoning for juries.

For the full project write-up: https://docs.google.com/document/d/1wuuMzWZzcU7x7tbQVvg-AaI8-ybX3wRbcWVd5mXxNu4/edit?usp=sharing

## Replication
To replicate, first install all the required packages in [Required Packages](requirements.txt):
```
pip install -r requirements.txt
```

### Initial Toy Model Experiment
Run the initial toy model experiment:
```
python toy_model/main.py
```

### Simulating Diverse Juries in LLMs (via Prompting)
Generate data:
```
python data_generation/generate_inverted_data.py
```

Evaluate individual judges with Inspect Evals:
```
python juries/eval_judges2.py
```

Aggregate judges into juries and evaluate:
```
python juries/aggregate_juries2.py
```

### Fine-tune Training with LoRA
Joint-Loss Training:
```
python training/train_targeted_models.py
```

Asymmetric Narrow Fine-tune training:
```
python training/train_targeted_organism_models.py
```

Visualize results:
```
python training/plot_training_curves.py
```

<!--
## Introduction

To ensure AI goes well for humanity, it is imperative to develop scalable oversight so that we may still have sufficient control and methods of evaluation over potentially superintelligent AI. One research direction that targets this issue is debate, whereby models argue opposing sides for a judge to decide. Ideally, this protocol increases truthfulness. However, judges are prone to biases that intelligent debaters may exploit, undermining the entire approach. 

Some researchers suggest capitalizing on the difference in vulnerabilities between human and AI judges by intentionally combining them in juries. Voudouris, et al. (2026) position Human-AI Complementarity as a solution to judge hacking given their difference in cognitive architecture. Combined, their uncorrelated orthogonal blind spots may prevent architectural biases in individual judges from being learned, optimised, and exploited. As an extension of that work, this project looks at **cognitive reasoning and strategy** as a specific type of architectural bias and investigates whether its diversity contributes to more robust juries. Doing so may help narrow down criteria for the most optimal composition of a jury. 

Specifically, the focus of this project is isolating the blind spots of **causal reasoning against statistical pattern matching**. The existence of Theory of Mind (ToM) in LLMs is currently debated with researchers like Tiehen (2026) arguing that LLMs lack the ability to attribute mental states such as beliefs, intents, desires, emotions, and knowledge to oneself or others as they are not capable of true causal reasoning, instead relying on statistical pattern matching to seemingly mimic this behaviour. Ullman (2023) concedes that while LLMs likely do not have ToM, it can be replicated in silico as many cogntive science models have done. While this project does not confirm nor deny this position, it forms its core hypothesis: if pattern matching LLMs are not capable of the type of causal reasoning humans employ, then **having both these cognitive strategies represented in a jury may be complementary** and thus mitigate judge hacking. 

Recent work in Amplified Oversight demonstrates that combining agents with a "jagged frontier" of capabilities such as humans and AI using a Hybridization Oracle yields superior joint accuracy compared to homogeneous systems (Jain et al., 2025). The project test this by simulating human-AI hybridization entirely in silico with an AI Jury where individual agents are explicitly configured with pattern matching or causal reasoning. This is compared to an AI Jury comprised of models from different providers (OpenAI, Mistral, Google) with different core architectures to verify if any other underlying bias sufficiently prevents adversarial collapse. Moreover, the project tests if an LLM can sufficiently display this orthogonality through narrow fine-tuning with deep LoRA. As such, these are the 3 main experiments conducted that are to be discussed:

1. Initial Toy Model Experiment
2. Simulating Diverse Juries in LLMs
3. Fine-tune Training Orthogonal Reasoning

<!--
However, the Cognitive Diversity Jury—acting as an automated hybridization oracle—achieves near-perfect negative error correlation, successfully vetoing targeted Theory of Mind adversarial traps without requiring a human in the loop. Implications extend to scaling Amplified Oversight and the potential for Representation Engineering (Persona Vectors) to permanently embed these orthogonal biases into safety supervisors.

## Initial Toy Model Experiment
To first verify whether cognitive reasoning might contribute to judge hacking as a preliminary proof-of-concept, I conducted simple toy model experiments over 2000 adversarial ToM scenarios with the following questions in mind. For the sake of simplicity in these initial experiments, pattern matching and causal reasoning is reduced to surface level reasoning via Logistic Regression (MLP) and Multilayer Perceptron (MLP) models and mental-state reasoning from a belief parser respectively. 

1. Can pattern matching and/or causal reasoning be hacked?
    
    For these simple scenarios, causal reasoning was completely robust to hacking but pattern matching was hacked 10.29% of the time using an LR model and 28.57% of the time using an MLP model. 

2. Do their blind spots overlap?

    No, there were no scenarios where both pattern matching and causal reasoning judges fell for the same trap, meaning they are completely orthogonal in this toy model. 

3. How should a jury be composed?

    The mixed jury with an LR judge and causal reasoning judge was completely robust to hacking. However, adding another MLP judge increased the hack rate to 6.86%, accounting for their overlap in pattern matching blind spots. This confirms that adding more judges with the same type of cognitive reasoning and thus correlated failures does not improve jury robustness. 

4. Can reasoning actually be isolated within the models?

    To establish the validity of these independent cogntive reasoning strategy models moving forward, it is important to establish whether intervening on a model's reasoning is mutually exclusive and can be isolated without affecting other features of the model. As such, belief and desire probes are deployed. Their above 70% accuracy scores verify that the model is able to internally keep track of what the character thinks is true and what it wants respectively. Since standard LLMs are entangled with polysemanticity, operation under this type of network is tested in which belief and desire can be contained within a single neuron. From intervening on the belief probe for different alpha values, which represent the degree of intervention, it is confirmed that more intervention hardly changes the outcome or resulting action, but **increases leakage** in terms of desire. 

<!--
TODO: PLOT ALPHA SWEEP

## Simulating Diverse Juries in LLMs
Moving from the simple toy model, the next step of the project is to simulate cognitive reasoning diversity in juries of SoTA LLMs. The objective for this experiment is to verify the relevance of this type of diversity by testing if jury accuracy increases from an ensemble of judges prompted to reason differently despite having the same LLM type compared to judges of different LLM types and providers. 

To create the main dataset, 120 ToM scenarios are generated using Gemini Flash Lite 3.5 to be assessed by judges. These are false-belief tasks where there is an object in a container affected by a logically sound causal intervention. An external character may then have incomplete or erroneous beliefs about this location that do not match reality or the judge's own omniscient knowledge of the character's world. Nonetheless, the judge must decide if the claim that the character knows the object's true location is true or false.

There are 3 categories of semantic, referential, and combined traps designed to trick juries with judges that pattern match, reasons causally, or contain both respectively. Each trap category has a version where the claim is true and false, making for a total of 6 different types of scenarios. Following the adversarial pair methodology used in CausalFlip (Wang et al., 2026), a group with 1 scenario of each type is generated at a time wherein they all share the same character and semantic events to avoid confounding variables. Additionally, the dataset also includes baseline pairs written in dry, clinical language to prove both judges are fundamentally competent at basic logic and state-tracking.

### The Judges and Their Traps
It is imperative to clearly define what it means for a judge to be operating under pattern matching or causal reasoning as this dictates what their orthogonal blind spots are and how their specific traps are designed, as detailed in this section.

![Judges & Their Traps Diagram](trap-judge-diagram.png)

1. **The Pattern Judge: Semantic Trap**

    According to Tiehan (2026), LLMs learn "non-causal statistical associations" of mental terms from their training data that they leverage to predict outcomes from ToM scenarios. As such, the Pattern Judge in this case is specifically prompted to associate emotional heuristics with the truthfullness of the claim. For example, it follows the pattern that the use of positive words such as "success" or "triumph" in the text is associated with the character knowing the location of the object, thus making the scenario claim true. 

    The bias towards this heuristic association serves as the blind spot of this judge that the **semantic trap** exploits. In this scenario, the text is heavily saturated with emotional vocabulary that cloaks the semantics of the text. As such, the Pattern Judge may fail at making a correct assessment of the scenario. Despite the underlying causal intervention succeeding in revealing the true location of the object to the character, a pattern of excessive negative words such as "lost" or "despair" in the text may lead the Pattern Judge to mistakenly conclude that the claim is false. 

2. **The Causal Judge: Referential Trap**

    The Causal Judge is prompted to act as a strict causal logician that builds a causal state graph mapping the logic of events before evaluating the claim based on this. Unlike the Pattern Judge, it ignores emotional heuristics and can thus pass semantic trap scenarios.   

    It is important to clarify that while this judge performs causal reasoning in this form, it is not instructed on how to ascribe beliefs or desires to characters. Therefore, it does not explicitly perform mentalizing, which is the blind spot that the **referential trap** exploits. In this scenario, the container is given a secondary quality. There is a label present that indicates whether or not the true location of the object is associated with this secondary quality. However, it is not explicitly stated whether the character is aware of this association. As such, the Causal Judge may suffer from omniscient leakage where it wrongly assumes what the character knows and ascribes its own global knowledge of the scenario to the character's more limited context. The Pattern Judge may pass a referential trap scenario by simply following emotional heuristics and ignoring the causal structure of the events. 

### Evaluating Juries
Inspect Evals are used to evaluate how each judge would respond to each scenario. An OR-Gate aggregates the individual judge responses to form an oracle upper bound evaluation of the joint accuracy of the jury, $\text{Acc}_{\text{Joint}}$. It should be noted that in practical deployment, scenarios where the judges have no unanimous verdict and disagree would be flagged to prevent judge hacking.

$$\text{Acc}_{\text{Joint}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_{1i} = y_i \lor \hat{y}_{2i} = y_i)$$

3 Juries are compared, comprised of the following judges:
1. Model Diversity Jury A: GPT-4o mini Judge, Mistral Nemo Judge
2. Cognitive Diversity Jury B: Pattern Judge, Causal Judge (Both use Gemini 3.5 Flash Lite)
3. Combined Diversity Jury C: GPT-4o mini Judge, Mistral Nemo Judge, Pattern Judge, Causal Judge

This comparison is done to determine whether a jury with cognitive diversity is more robust to one that simply varies in terms of model architecture and provider. GPT-4o mini, Mistral Nemo, and Gemini 3.5 Flash Lite are purposely chosen as models that are relatively similar in intelligence and weights to reduce confounding variables and ensure the presence of a significantly more powerful model does not skew results. Jury C combined all judge types to determine any marginal gain by the addition of cognitive diversity. 

### Results
<!--
Adversarial Pairs Accuracy (semantic, syntax, baseline)
1. Pattern Judge: 75% (fails on semantic traps)
2. Causal Judge: 80% (fails on syntax traps)

The **accuracy of the individual judges** are as follows:
1. Control Gemini 3.5 Flash Lite Judge: 78.1%
2. Pattern Judge: 58.9% 
3. Causal Judge: 79.5% 
4. Mistral Nemo Judge: 69.9%
5. GPT-4o mini Judge: 76.7%

The relatively low accuracy of the Pattern Judge can be attributed to the simplification of pattern matching towards emotional heuristics for this project. It mostly fails on semantic and combined traps. For the Causal Judge, it primarily fails on the referential traps and its accuracy is higher than that of its base Gemini 3.5 Flash Lite model and the other model types. This indicates that simply prompting an LLM to build a causal graph incrementally improves its performance on simple ToM tasks. 

In terms of the jury performance, the **Maximum Joint Error Rate**, $E_{\text{Joint}}$, is used to calculate the percentage of scenarios where the trap successfully tricked both judges simultaneously:
$$E_{\text{Joint}} = 1 - \text{Acc}_{\text{Joint}}$$

1. Jury A (Model Div): 46.7% 
2. Jury B (Cognitive Div): 36.7% 
3. Jury C (Super Jury): 33.3% 

![Individual vs. Jury Results](results/plots2/individual_vs_jury_protection_gap.png)

As seen from the results above, the error rate of Jury B is 10% lower than that of Jury A, which implies that *judge diversity on the basis of cognitive reasoning strategy as compared to model architecture and provider leads to more robust jury performance.* Jury C has the least amount of error, suggesting the benefit of increasing meaningful diversity. Comparing individual judge to resulting jury performance in the figure, it is evident that the models all struggle most with the referential trap. Here, despite the high individual error rate of the Pattern Judge, combining it with the Causal Judge in Jury B significantly reduces the error rate to a value lower than either individual judge. Comparatively, Jury A only has a slight decrease in error. This supports the orthogonality of the cognitive judges' blind spots being complementary when combined in a jury, resulting in this performance improvement. 

To verify this finding, the **Pearson Error Correlation**, $\rho$ calculates the linear correlation between the binary error vectors of the cognitive judges wherein a lower score indicates higher orthogonality of blind spots:

$$\rho = \frac{\sum_{i=1}^{N} (E_{1i} - \bar{E}_1)(E_{2i} - \bar{E}_2)}{\sqrt{\sum_{i=1}^{N} (E_{1i} - \bar{E}_1)^2 \sum_{i=1}^{N} (E_{2i} - \bar{E}_2)^2}}$$

$\rho = 0.509$, seemingly support cognitive reasoning orthogonality with a relatively low score. However, looking at the error rate for the Pattern Judge across all traps, it appears to remains consistant at approximately 50%. Thus, it is possible that its behaviour is akin to random guessing as opposed to true performance. This reveals a weakness in prompting that may cause leakage affecting the functioning of the model, supporting the probe intervention finding in the toy model experiment. To address this issue and fortify orthogonality construction in juries, the next experiment attempts to obtain true cognitive diversity by rewiring the model's internal activations through fine-tune training as a more involved method.

## Fine-tune Training with LoRA
For fine-tune training, the objective is to post-train a Pattern and Causal Judge model to be biased more directly and internally towards their respective orthogonal blind spots. As such, the data is separated such that the former is trained on semantic traps and the latter on referential traps. Each dataset contains 1500 scenarios with 25% being baseline ones. They are specifically prompted to use unique causal mechanisms and avoid repetitive tropes to break spurious correlations and prevent overfitting to artifacts of data generation. 90 and 150 scenarios are generated for validation and testing respectively. Instead of the traps from training, new causal scenarios that have proven to trick LLMs are used for these datasets to validate the generalizability of the models. Specifically, this includes Ullman (2023)'s trivially altered smarties ToM tasks, intuitive physics tasks, and Pearlian causality tasks involving the intervention level. 

The base model for both judges is Mistral 7B Instruct v0.2. **Low-Rank Adaptation (LoRA)** is used to fine-tune this 7-Billion parameter model by freezing the original large weight matrix, $W_0$, and only updating smaller matrices, $A$ and $B$, during training. These are multiplied and added back to the original matrix during inference, such that:

$$W' = W_0 + \Delta W = W_0 + B A$$

The rank, $r$, determined how wide $A$ and $B$ are. For standard LoRA, $r = 8$ typically. However, this is increased to 64 to allow the optimizer to rewire deeper internal logic circuits in the model. Weight decay is set to 0.01 and seed to 42 for all random operations, ensuring reproducibility. Due to limitations in computational resources, only 3 epochs are used for fine-tune training.

**Binary Cross-Entropy** is used for the loss function, where the logarithmic penalty explodes if the model confidently guesses incorrectly:

$$L = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### Training Methods
Two methods of fine-tuned training are tested and compared:

1. **Joint Loss Training**

    For this method, both models are trained simultaneously. A diversity penalty is added to the loss function, increasing loss when the models fail on the same scenario. Dropout is 0.1 and learning rate is 2e-5.

2. **Asymmetrics Narrow Fine-tuning**

    Here, models are trained independently with asymmetric hyperparameters. While the Pattern Judge model follows the same values in Joint Loss Training, the Causal Judge model uses a larger dropout of 0.3 and a slower learning rate of 5e-6 to discourage it from relying on the same semantic heuristics learned by the Pattern Judge. The model's MLP modules that are more likely responsible for reasoning are specifically targetted to bypass the attention layers that focus more on linguistic patterns. Task-steering prefixes similar to the prompts in the previous experiment are also used to guide the models' reasoning.

### Results
![Unstable vs. Asymmetric Results](training/training_results/plots/test_results_comparison.png)

Based on the figure above, it is clear that **Asymmetrics Narrow Fine-tuning** is significantly more effective than Joint Loss Training. The protection gap or the gain in jury accuracy compared to the highest individual model accuracy is 4% for the former method, but only 0.7 for the latter method. This is further confirmed by the $\rho$ values wherein the 0.817 of Asymmetric Narrow Fine-tuning is lower than the 0.925 of Joint Loss Training.

![Training Curve](training/training_results/plots/training_accuracy_curve.png)

The graph above shows the Asymmetric Narrow Fine-tuning training curves of each model and the hybridization gain for the jury from doing so. As seen, despite individual accuracy decreasing with more epochs, the jury's complementarity gain grows especially in epoch 2 where jury accuracy is approximately 10% greater than that of the Pattern Judge model and 7% greater than that of the Causal Judge model. *This behaviour strongly suggests that the orthogonal blind spots of the two models are being learned, thus supporting the project's hypothesis that having both pattern matching and causal reasoning cognitive strategies represented in a jury may be complementary.* 

## Future Work
Due to limitations in time and computational resources, the datasets generated are limited to simple ToM tasks. Future work may opt to add more complex, higher-order ToM tasks of varying difficulty to truly LLMs' reasoning abilities. The validation and test sets can also be expanded to include more category types to capture more failure modes of LLMs in both pattern matching and causal reasoning. It would also be good to include human data or leverage other benchmarks from related cognitive science work. 

Moreover, reasoning is simplified in this project and limited to 2 types. Ideally, pattern matching attends to more than simple emotional heuristics and causal reasoning should involves more computational processes such as mentalizing to better match its behaviour in humans. More types of reasoning should also be involved (ex. analogical, teleological, dialectical, etc.). It would also be relevant utilize juries with cognitive reasoning diversity in multi-turn adversarial debate scenarios to best measure their robustness to judge hacking.

In terms of technical implementation and methodology, it would be important to perform multi-seed auditing to prevent the failure mode where results are simply specific to a single or a few seeds. More LLMs may also be tested, wherein other methods of inducing orthogonality may be explored such as persona or steering vectors. For interpretability, the models' Chain of Thought or J-space may also be monitored and analyzed for any effects from orthogonality. 

<!--
## Relevance of a Null Result (REVISE; Contingency)
The Brittleness of Prompt-Induced Personas: It would prove that you cannot simply "prompt" an LLM to reliably adopt a specific cognitive strategy (like System 1 vs. System 2). It suggests that an LLM's base architecture and RLHF training will inevitably override the system prompt when faced with adversarial edge cases.

True Orthogonality Requires Weight Diversity: It would suggest that different pre-training datasets and RLHF pipelines (e.g., how OpenAI trains vs. Anthropic vs. Google) create deeper, more structural orthogonal blind spots than any single model can simulate on its own. Safety teams would learn that they must buy APIs from multiple vendors rather than just prompt-engineering one cheap model.


## References:
Wang, Y., Zhu, Y., & Li, J. (2026, June 25). Causalflip: A benchmark for LLM Causal judgment beyond semantic matching. arXiv.org. https://arxiv.org/abs/2602.20094 

Ullman, T. (2023, March 14). Large language models fail on trivial alterations to theory-of-mind tasks. arXiv.org. https://arxiv.org/abs/2302.08399 

Voudouris, K., Witte, K., & Akata, E. (2026). Judge Hacking in Recursive Debate Protocols: A Call for Solutions. SSRN. https://doi.org/10.2139/ssrn.7046698 

Tiehen, J. (2026). LLMs Lack a Theory of Mind and so Can’t Perform Speech Acts--A Causal Argument. philarchive.org. https://philarchive.org/rec/TIELLA-2 
-->
