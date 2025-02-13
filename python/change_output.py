import neat
import pandas as pd
import random
import numpy as np
import pickle

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
neat.Config.random_seed = 0

# Load the dataset
data = pd.read_csv("USDJPY_D1_5years.csv")

# Define the trading environment
class TradingEnvironment:
    def __init__(self, data, leverage=1, spread_rate=0.0, capital=10000):
        self.data = data
        self.index = 0
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.leverage = leverage
        self.spread_rate = spread_rate
        self.capital = capital

    def get_state(self):
        if self.index >= len(self.data):
            return None
        trendline = self.data.iloc[self.index]['trendline']
        price_diff = self.data.iloc[self.index]['price_diff']
        volume = self.data.iloc[self.index]['tick_volume']
        close = self.data.iloc[self.index]['close']
        time_sin = self.data.iloc[self.index]['time_sin']
        time_cos = self.data.iloc[self.index]['time_cos']
        return [trendline, price_diff, volume, self.position ]

    def step(self, action):
        reward = 0
        profit = 0

        if action == 1 and self.position == 0:  # Open long position
            self.position = 1
            self.entry_price = self.data.iloc[self.index]['close']
        elif action == 2 and self.position == 1:  # Close long position
            raw_profit = (self.data.iloc[self.index]['close'] - self.entry_price ) * self.leverage * (self.capital / self.entry_price) 
            #raw_profit = (  self.entry_price - self.data.iloc[self.index]['close']) * self.leverage * (self.capital / self.entry_price)
            spread_cost = abs(raw_profit) * self.spread_rate
            profit = raw_profit - spread_cost
            self.capital += profit
            reward += profit
            self.position = 0
        elif action == 3 and self.position == 0:  # Open short position
            self.position = -1
            self.entry_price = self.data.iloc[self.index]['close']
        elif action == 4 and self.position == -1:  # Close short position
            raw_profit = (  self.entry_price - self.data.iloc[self.index]['close']) * self.leverage * (self.capital / self.entry_price) 
            #raw_profit = (self.data.iloc[self.index]['close'] - self.entry_price ) * self.leverage * (self.capital / self.entry_price)
            spread_cost = abs(raw_profit) * self.spread_rate
            profit = raw_profit - spread_cost
            self.capital += profit
            reward += profit
            self.position = 0
        elif action == 0:  # Do nothing
            reward -= (self.capital * 0.01)  # Penalize inaction

        
        # Update state index
        self.index += 1
        return self.get_state(), reward, profit

# Softmax function for action probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Fitness evaluation function
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    train_data = data[:int(0.8 * len(data))]
    env = TradingEnvironment(train_data)
    total_profit = 0
    total_reward = 0
    actions_log = []
    long_trades = 0
    short_trades = 0
    for _ in range(40):
        state = env.get_state()
        if state is None:
            break
        # raw_outputs = net.activate(state)  # Raw scores from the neural network
        # probabilities = softmax(raw_outputs)
        # action = np.random.choice(len(probabilities), p=probabilities)  # Sample actions probabilistically
        action = np.argmax(net.activate(state))
        _, reward, profit = env.step(action)

        total_reward += reward
        total_profit += profit
        if action in [1, 2]:  # Long trades
            long_trades += 1
        elif action in [3, 4]:  # Short trades
            short_trades += 1

        actions_log.append({
            "index": env.index - 1 + 1,
            "state": state,
            "action": action,
            "reward": reward,
            "profit": profit,
            "capital": env.capital,
            'close': env.data.iloc[env.index]['close']
        })

    # Penalize imbalanced behavior
    balance_penalty = -abs(long_trades - short_trades) * 0.1
    genome.fitness = total_reward + balance_penalty

    return total_reward, total_profit, actions_log

# NEAT configuration
def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT for 20 generations
    winner = population.run(fitness_function, 20)

    # Log the best genome's performance
    _, _, actions_log = evaluate_genome(winner, config)
    for log_entry in actions_log:
        log_line = (
            f"Step {log_entry['index']}:\n"
            f"  State: {log_entry['state']}\n"
            f"  Action: {log_entry['action']}\n"
            f"  Reward: {log_entry['reward']:.2f}, Profit: {log_entry['profit']:.2f}, "
            f"Capital: {log_entry['capital']:.2f}\n"
            f"Close: {log_entry['close']:.2f}\n"
        )
        print(log_line)
    bestgenome_action_counts = {"long": 0, "short": 0, "hold": 0}
     # Count actions from the genome's logs
    for log_entry in actions_log:
        action = log_entry["action"]
        if action in [1, 2]:  # Long actions (1 = open, 2 = close)
            bestgenome_action_counts["long"] += 1
        elif action in [3, 4]:  # Short actions (3 = open, 4 = close)
            bestgenome_action_counts["short"] += 1
        elif action == 0:  # Hold action
            bestgenome_action_counts["hold"] += 1
    
    print(f"\nBestgenome Action Counts:\n{bestgenome_action_counts}\n")

    # Save the best genome
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:\n", winner)
    print("\nProfit:\n", winner.fitness)


# Fitness function wrapper for NEAT
def fitness_function(genomes, config):
    global_action_counts = {"long": 0, "short": 0, "hold": 0}

    for genome_id, genome in genomes:
        total_reward, total_profit, actions_log = evaluate_genome(genome, config)

        # Count actions from the genome's logs
        for log_entry in actions_log:
            action = log_entry["action"]
            if action in [1, 2]:  # Long actions (1 = open, 2 = close)
                global_action_counts["long"] += 1
            elif action in [3, 4]:  # Short actions (3 = open, 4 = close)
                global_action_counts["short"] += 1
            elif action == 0:  # Hold action
                global_action_counts["hold"] += 1
    # Print action counts for this generation
    print(f"\nGeneration Action Counts:\n{global_action_counts}\n")


# Main entry point
if __name__ == "__main__":
    # NEAT config file path
    config_path = "neat-config.txt"
    run_neat(config_path)
