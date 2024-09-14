---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Credit Adjustment

Now, we'll use reinforcement learning (RL) to dynamically adjust credit limits and interest rates over time based on repayment behavior.

2.1. Data Structure for RL
You will need:
 * State: Customer features and repayment history.
 * Actions: Adjusting credit limit and interest rate.
 * Reward: Profit (as described earlier).

2.2. Deep Q-Learning (DQN) Agent

You can now use Deep Q-Learning (DQN) to train the RL agent to adjust credit limits and interest rates. A neural network is used as the Q-function approximator to estimate the reward for each state-action pair.

1. Setting up DQN with Stable Baselines3
To train a DQN agent to dynamically adjust credit limits and interest rates based on customer repayment behavior, we need to wrap our custom environment (CreditCardEnv) and set up the DQN algorithm. Here's the continuation:

```
# Install necessary package for RL (Stable Baselines3)
!pip install stable-baselines3

# Import necessary packages
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv

# Define the environment (Credit Card Environment)
class CreditCardEnv:
    def __init__(self, customer_features, repayment_history, max_credit_limit=20000):
        self.customer_features = customer_features
        self.repayment_history = repayment_history
        self.max_credit_limit = max_credit_limit
        self.credit_limit = 0
        self.interest_rate = 0.0
        self.time_step = 0
        self.max_time_step = 24  # Simulate 24 months
        
    def reset(self):
        self.time_step = 0
        self.credit_limit = random.randint(1000, self.max_credit_limit)
        self.interest_rate = random.uniform(0.05, 0.25)
        return self._get_state()
    
    def _get_state(self):
        return np.array([self.customer_features, self.repayment_history, self.credit_limit, self.interest_rate])
    
    def step(self, action):
        """
        Action could be adjusting the credit limit or interest rate.
        Simulate repayment behavior based on action.
        """
        self.time_step += 1
        
        # Simulate repayment outcome (use random behavior for now)
        repayment_made = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% chance of repayment
        
        # Calculate reward (profit)
        revenue = self.credit_limit * self.interest_rate * repayment_made
        cost = self.credit_limit * (1 - repayment_made)
        profit = revenue - cost
        
        # Adjust the credit limit or interest rate based on the action
        # (Simplify action space for now, e.g., action=0 means no change, action=1 means increase interest rate)
        if action == 1:
            self.interest_rate = min(self.interest_rate + 0.01, 0.25)
        elif action == 2:
            self.credit_limit = min(self.credit_limit + 1000, self.max_credit_limit)
        
        # Done if we've reached the final month
        done = self.time_step >= self.max_time_step
        
        return self._get_state(), profit, done
    
# Initialize environment with some sample data
env = CreditCardEnv(customer_features=np.random.rand(5), repayment_history=[1, 1, 0])

# Wrap the environment in a vectorized environment to support parallel simulation
env = DummyVecEnv([lambda: env])

# Initialize the DQN agent
model = DQN('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("credit_card_dqn_model")

# Load the model to use later
model = DQN.load("credit_card_dqn_model")

# Test the trained agent
state = env.reset()
done = False

while not done:
    action, _ = model.predict(state)
    state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
```

Explanation of the Code:
1. Environment: We use the `CreditCardEnv` to simulate a real-world scenario of dynamically adjusting credit limits and interest rates based on customer repayment behavior.
2. Actions: Actions can include increasing interest rates or adjusting the credit limit, and the agent will learn to take these actions to maximize long-term profit.
3. Reward: The reward is calculated as the **profit**, which is the difference between the revenue from repayments and the cost of defaults.
4. DQN Agent: We use the `DQN` algorithm from `Stable Baselines3`, which is a Deep Q-Learning agent, to learn an optimal policy for maximizing the reward (profit) over multiple time steps (e.g., 24 months).

Next Steps for RL

1. Tune Hyperparameters: You can experiment with different learning rates, exploration/exploitation rates, and batch sizes to improve the performance of the RL agent.
2. Action Space: Expand the action space to allow more granular adjustments to credit limits and interest rates. For example, you can allow both increments and decrements for interest rates and credit limits.
3. Advanced RL Algorithms: If you want to explore more advanced RL techniques, you can try algorithms like DDPG (Deep Deterministic Policy Gradient) for continuous action spaces or PPO (Proximal Policy Optimization).