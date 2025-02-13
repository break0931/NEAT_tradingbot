import neat
import pandas as pd
import numpy as np


data = pd.read_csv("XAUUSD_16408_6years.csv")

print(data)



# Simulated environment
class TradingEnvironment:
    def __init__(self, data, initial_capital=10000, leverage=1, spread_rate=0):
        self.data = data
        self.index = 0
        self.position = 0          # 0 = not holding, 1 = holding
        self.reward = 0
        self.capital = initial_capital
        self.leverage = leverage
        self.spread_rate = spread_rate
    def get_state(self):
        if(self.index >= len(self.data)):
            return None
        
        trendline = self.data.iloc[self.index]['trendline'] 
        price_diff = self.data.iloc[self.index]['price_diff']
        volume = self.data.iloc[self.index]['tick_volume']
        close = self.data.iloc[self.index]['close']
        time_sin = self.data.iloc[self.index]['time_sin']
        time_cos = self.data.iloc[self.index]['time_cos']
       
        return [trendline, price_diff, volume, self.position ]

    def step(self, action):
        """
        Actions:
        0 - Do nothing
        1 - Open trade
        2 - Close trade
        """
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
            reward = profit
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
            reward = profit
            self.position = 0
        elif action == 0:  # Do nothing
            reward -= (self.capital * 0.01)  # Penalize inaction
        # Update state index
        self.index += 1
        return self.get_state(), reward, profit
        
    


def test_best_genome(best_genome, config, test_data):
    # Create the neural network for the best genome
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # Initialize the environment with test data
    env = TradingEnvironment(test_data)
    total_profit = 0
    total_reward = 0

    while True:
        state = env.get_state()

        if state is None:  # Stop if the state is invalid
            break
        # epsilon = 0.2
        # if np.random.rand() < epsilon:
        #     action = np.random.choice(len(net.activate(state)))  # Random exploration
        # else:
        #     action = np.argmax(net.activate( state))
        print(net.activate(state))
        action = np.argmax(net.activate(state))  # Get action (0, 1, or 2)
        _, reward, profit = env.step(action)
        total_reward += reward
        total_profit += profit

    # Print and save the log
    # print("\n=== Test Log ===")
    # for entry in env.log:
    #     print(entry)

    print(f"\nFinal Capital: {env.capital}")
    print(f"Total Profit: {total_profit}")
    print(f"Total Reward: {total_reward}")

    # Optionally save the log to a file
    # with open("test_log.txt", "w") as log_file:
    #     for entry in env.log:
    #         log_file.write(entry + "\n")

if __name__ == "__main__":
    # Load the NEAT configuration
    config_path = "neat-config.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the best genome
    with open("best_genome_argmax_0-2.pkl", "rb") as f:
        import pickle
        best_genome = pickle.load(f)

    # Load unseen test data
    test_data = data[383:748]  # Last 20% of the dataset
    print(test_data)

    # Test the best genome
    test_best_genome(best_genome, config, test_data)



