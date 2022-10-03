# Reinforcement-Learning-Automatic-Trading - TFv2

We utilized reinforcement learning to perform automatic trading. Here, we implemented Deep Q-learning (DQL).
The project is implemented in tensorflow (TF) version 2.

There are several limitations in this project:
- We can only train and test on a single stock at a given time
- We can only perform one action per day (HOLD, BUY, or SELL)
- We can only buy or sell one single unit a day
- The simulation can end with units in the inventory
- The model is optimized based on the optimizing the actions, rather than maximizing the reward (profit)

The project is editted from:

https://www.mlq.ai/deep-reinforcement-learning-for-trading-with-tensorflow-2-0/

We noticed one issue. 

The model is optimized based on the rewards, which is only computed when a sell action is performed. Consequently, the reward table do not get adjusted properly when the hold and buy action is performed. We attempt to fix this by considering the following.

Assuming we hold or buy more stocks, and we already had some stock in our inventory of stock:
- If the oldest inventory stock price > the current stock price:
  - Holding or buying more is the right choice, and hence the reward is +0.5 (+ve)
  
- Elif the oldest inventory stock price < the current stock price
  - We should have sold the inventory stock, and hence the reward is -0.5 (-ve)

Additionally, we normalize the reward by applying a sigmoid function to the rewards, then substracting 0.5 to ensure that the range of the reward is [-0.5,0.5]. This is to avoid the reward difference. As the sell reward is equals to the profit of the sale, the sell reward can be extremely big (>100) or small (<-100). Consequently, the impact of the rewards due to sell versus those of hold/buy is significantly different. To ensure that the system learn properly, and not placing additional weights to the sell action, we applied the sigmoid function to normalize all the rewards.






