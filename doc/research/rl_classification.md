# Has reinforcement learning been used directly for classification?

Question from the advisor meeting (29 Oct 2026). Short answer: yes, but in
two specific forms, neither of which replaces a supervised classifier for
our problem. The project keeps supervised classifiers (GBM, MLP, GRU) at
the fusion centre and uses RL for coordination only.

## 1. The classifier itself as an RL agent

| Work | How RL is used | Relevance to us |
|---|---|---|
| E. Lin, Q. Chen, X. Qi, "Deep reinforcement learning for imbalanced classification", *Applied Intelligence*, 2020, doi:10.1007/s10489-020-01637-z (preprint arXiv:1901.01379) | Classification is posed as a sequential decision process: at each step the agent (a DQN) labels one sample and receives a reward; minority-class samples carry larger rewards. The learned policy *is* the classifier. | Mainly a tool against class imbalance. With labelled data a supervised loss reaches the same goal more directly (we use class-balanced training sets instead). |

## 2. RL decides *when* to classify (early classification of sequences)

| Work | How RL is used | Relevance to us |
|---|---|---|
| C. Martinez, G. Perrin, E. Ramasso, M. Rombaut, "A deep reinforcement learning approach for early classification of time series", EUSIPCO 2018; extended as "Adaptive early classification of temporal sequences using deep reinforcement learning", *Knowledge-Based Systems*, 2019 | A DQN agent observes a growing sequence and chooses either "wait" or "classify as class k"; the reward trades accuracy against earliness. | Closest to radar tracks: a track grows over time and an early but reliable class is valuable. |
| T. Hartvigsen, C. Sen, X. Kong, E. Rundensteiner, "Adaptive-Halting Policy Network for Early Classification" (EARLIEST), KDD 2019 | A recurrent classifier is trained with a supervised loss; a separate policy network, trained with RL, decides when to stop reading the sequence and output the class. | Same split as our design: the classifier stays supervised, RL controls a decision around it. |

## 3. RL for sensing decisions that serve classification (radar)

| Work | How RL is used | Relevance to us |
|---|---|---|
| W. W. Howard, S. R. Shebert, B. H. Kirk, R. M. Buehrer, "Mode Selection and Target Classification in Cognitive Radar Networks", arXiv:2310.17006, 2023 | Radar nodes learn when to observe actively and when to estimate passively, while class distributions are learned; RL selects the *sensing mode*, classification follows. | Supports our design: RL coordinates the network (which radar observes what), classification is done from the resulting evidence. |

## Conclusion for the project

* We did not find radar target-type classification where an RL agent is
  the classifier and outperforms supervised models on labelled data.
* RL is used around classification: to handle imbalance, to decide when
  the evidence is enough (early classification), or to choose sensing
  actions that improve the evidence.
* Our design follows the last two ideas: the fusion-centre RL policy
  coordinates the radars for track continuity; classification is done by
  supervised models on the fused track. An "early, confident class" reward
  term (as in Martinez et al.) is a possible later extension.

These references were checked against publisher / arXiv records in
November 2026; the full texts were not all available, so details beyond the
abstracts should be checked before citing them in the report.
