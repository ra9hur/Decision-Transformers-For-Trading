## Decision Transformers For Trading

DRL framework is powerful in solving dynamic decision making problems by learning through interactions with an unknown environment, thus exhibiting two major advantages: portfolio scalability and market model independence. 

Automated trading is essentially making dynamic decisions, namely to decide where to trade, at what price, and what quantity, over a highly stochastic and complex stock market. Taking many complex financial factors into account, DRL trading agents build a multi-factor model and provide algorithmic trading strategies, which are difficult for human traders.

<p align="center">
<img src="https://github.com/user-attachments/assets/4d8149a7-5dae-4197-9b48-108be64a30e0" alt="image" width="600" height="auto">
</p>

However, current algorithms for reinforcement learning are online based, involves direct interaction with the environment, real-time data collection, and continuous learning. These are mostly based on dynamic programming, which involves following an inner loop optimization problem that's very unstable. The variance in the returns that we tend to get in RL is really huge even after multiple rounds of experimentation.

For better decision making, input data should be multi-modal. Since transformers already support texts, visual, audio (list is growing), provides stable training and then can be scaled as well, RL architectures based on transformers could be a good choice in the coming years.

#### Decision Transformers
The main idea is that instead of training a policy using RL methods, such as fitting a value function, that will tell us what action to take to maximize the return (cumulative reward), we use a sequence modeling algorithm (Transformer) that, given a desired return, past states, and actions, will generate future actions to achieve this desired return. It’s an auto-regressive model conditioned on the desired return, past states, and actions to generate future actions that achieve the desired return.

This is a complete shift in the Reinforcement Learning paradigm since we use generative trajectory modeling (modeling the joint distribution of the sequence of states, actions, and rewards) to replace conventional RL algorithms. This means that in Decision Transformers, we don’t maximize the return but rather generate a series of future actions that achieve the desired return.

<p align="center">
<img src="https://github.com/user-attachments/assets/41e3bc10-2594-4c05-b48f-17666cc8fb1c" alt="image" width="600" height="auto">
</p>

Decision Transformers are offline RL based, involves training an agent using a pre-collected dataset without interacting with the environment. The agent learns from a fixed dataset, which can be collected from previous interactions, human demonstrations, or other sources.

#### FinRL
For trading and financial functionality, found FinRL very useful. FinRL is a open source framework for financial reinforcement learning. It provides a framework that supports various markets, SOTA DRL algorithms, benchmarks of many quant finance tasks, live trading, etc.

Existing FinRL framework that currently supports DRL algorithms such as PPO, A2C. In this prototype, the framework is extended to support Decision Transformers.

***

## Train And Evaluate

Summarizing the steps that were followed to train and evaluate Decision Transformer.

1. **Collect Dow Jones Index data** \
YahooDownloader is used to fetch 30 stocks in Dow Jones Index. We have used FinRL that provides a few excellent pre-processing packages to download and further pre-process the data.

2. **Prepare inputs to Decision Transformer** \
Here, dataset of experiences, which include states, actions, and rewards is prepared. Using the DJI downloaded data from the previous step, a gym-anytrading environment is created. A PPO based agent is used to generate trajectories that include (states, actions, rewards, done) information.

3. **Train** \
A data collator is used to sample trajectories of specified context lengths to ensure diverse training data. Training batches are created containing states, actions, rewards, returns, timesteps, and attention masks. \
In order to train the model with the trainer class, we first need to ensure the dictionary it returns contains a loss, in this case L-2 norm of the models action predictions and the targets. We achieve this by making a TrainableDT class, which inherits from the Decision Transformer model.

4. **Evaluate** \
The model's prediction is conditioned on sequences of states, actions, time-steps and returns. The action for the current time-step is included as zeros and masked in to not skew the model's attention distribution.
Tried setting,  <br />
TARGET_RETURN = 1.1 * (maximum reward in training dataset) \
Results did not change at all.

***

## Visualize and analyze performance

#### PPO Vs Decision Transformers
![PPO_VS_DT](https://github.com/user-attachments/assets/caa7d722-c72b-4123-9185-03b3640b581f)

PPO agent was used to generate trajectories which served as input to train Decision Transformers. This chart is not a comparison chart for Decision Transformer and PPO's performance. This is only to indicate that Decision Transformer has further learnt and is not just mimicking PPO predictions. 

#### Decision Transformers Vs Dow Jones Index
![DT_VS_DJI](https://github.com/user-attachments/assets/57918c30-a072-468b-ba46-65617f15c45d)

Here, Decision Transformer's predictions are slightly better OR atleast as good as the performance of Dow Jones Index.

#### Decision Transformers Vs Mean Variance Optimization Vs Dow Jones Index
![DT_VS_MVO_VS_DJI](https://github.com/user-attachments/assets/34c0e9a1-d039-4fe1-a5ff-594aadbd34c1)

Mean Variance Optimization computes co-variance among stocks, uses this information to create a portfolio in order to maximize returns with minimum risk. As seen in the chart, this clearly outperforms Decision Transformer and Dow Jones Index.

#### PPO Vs Dow Jones Index
![PPO_VS_DJI](https://github.com/user-attachments/assets/266458d0-5e49-4558-9ad1-38a5b16d2254)

No specific attempts were made to fine-tune PPO training parameters. With necessary changes, PPO predictions could get better.


## Improvements
1. As discussed in [Offline Reinforcement Learning: BayLearn 2021 Keynote Talk](https://www.youtube.com/watch?v=k08N5a0gG0A), offline learning followed by online fine-tuning is expected to yield better performance and this should be tried out.
2. Market sentiment is the current attitude of investors overall regarding a company, a sector, or the financial market as a whole. The mood of the market is affected by crowd psychology. It is revealed through buying and selling activity. This plays a huge role in predicting movement of stock prices and should be considered for predictions.


## References:

1. [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
2. [Offline Reinforcement Learning: BayLearn 2021 Keynote Talk](https://www.youtube.com/watch?v=k08N5a0gG0A)
3. [Stanford CS25: V1 I Decision Transformer: Reinforcement Learning via Sequence Modeling](https://www.youtube.com/watch?v=w4Bw8WYL8Ps)
4. [Decision Transformers: Eindhoven RL Seminar](https://www.youtube.com/watch?v=83QN9S-0I84)
5. [Train your first Decision Transformer](https://huggingface.co/blog/train-decision-transformers)
6. [FinRL Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials/tree/master?tab=readme-ov-file)
