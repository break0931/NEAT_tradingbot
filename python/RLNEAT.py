import neat
import pandas as pd
import numpy as np


data = pd.read_csv("GBPUSD_H1_5years.csv")

print(data)



# Simulated environment
class TradingEnvironment:
    def __init__(self,data):
        self.data = data
        self.index = 0
        self.holding = 0          # 0 = not holding, 1 = holding
        self.reward = 0

    def get_state(self):
        if(self.index >= len(self.data)):
            return None
        
        trendline = self.data.iloc[self.index]['trendline'] 
        price_diff = self.data.iloc[self.index]['price_diff']
        volume = self.data.iloc[self.index]['tick_volume']
        close = self.data.iloc[self.index]['close']
        time_sin = self.data.iloc[self.index]['time_sin']
        time_cos = self.data.iloc[self.index]['time_cos']
        self.index += 1
        return [trendline, price_diff, volume, close , time_sin , time_cos , self.holding ]

    def step(self, action):
        """
        Actions:
        0 - Do nothing
        1 - Open trade
        2 - Close trade
        """
        reward = 0
        profit = 0
        leverage = 100
        capital = 10000  # Fixed trade capital
        spread_rate = 0 

        if action == 1 and self.holding == 0:  # Open trade
            self.holding = 1
            self.entry_price = self.data.iloc[self.index]['close']

        elif action == 2 and self.holding == 1:  # Close trade
            self.holding = 0
            raw_profit = (self.data.iloc[self.index]['close'] - self.entry_price) * leverage * (capital / self.entry_price)
            spread_cost = abs(raw_profit) * spread_rate
            net_profit = raw_profit - spread_cost
            profit = net_profit
            reward = (self.data.iloc[self.index]['close'] - self.entry_price) # Scaled reward for learning

        elif action == 0:  # Do nothing
            reward = -0.01  # Penalize doing nothing

        return self.get_state(), reward, profit
        
    

# Fitness evaluation function
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    train_data = data[:int(0.8 * len(data))]
    env = TradingEnvironment(train_data)
    total_profit = 0
    total_reward = 0
    num_step = len(train_data)
    for _ in range(100):  # Simulate 100 steps
        state = env.get_state()
        if state is None:  # Stop if the state is invalid
            break
        action = np.argmax(net.activate(state))  # Get action (0, 1, or 2)  
        _, reward , profit = env.step(action)
        total_reward += reward
        total_profit += profit
        

    return total_reward , total_profit

# NEAT configuration
def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT for 50 generations
    winner = population.run(fitness_function, 20)

    # Save the best genome
    with open("best_genome.pkl", "wb") as f:
        import pickle
        pickle.dump(winner, f)

    print("\nBest genome:\n", winner)
    print("\nprofit:\n", winner.profit)
# Fitness function wrapper for NEAT
def fitness_function(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness , genome.profit = evaluate_genome(genome, config)

# Main entry point
if __name__ == "__main__":
    # NEAT config file path
    config_path = "neat-config.txt"

    
    run_neat(config_path)



