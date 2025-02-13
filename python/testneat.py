import neat
import pandas as pd
import numpy as np
import random

# Load historical forex data (example for EUR/USD)
data = pd.read_csv("./python/trendline.csv")

# Calculate additional features (Trendline, RSI, MACD, etc.)
# def calculate_rsi(data, window=14):
#     delta = data['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))

# def calculate_macd(data):
#     ema12 = data['close'].ewm(span=12, adjust=False).mean()
#     ema26 = data['close'].ewm(span=26, adjust=False).mean()
#     return ema12 - ema26

# data['rsi'] = calculate_rsi(data)
# data['macd'] = calculate_macd(data)
# # Calculate price difference between last two candles
# data['price_diff'] = data['close'].diff()




# Dummy outputs (for training purpose)
# data['open_trade'] = np.random.choice([0, 1], size=len(data))  # Randomly open trades
# data['close_trade'] = np.random.choice([0, 1], size=len(data)) # Randomly close trades


# Prepare inputs for NEAT model (5 inputs total)


data['holding'] = np.random.choice([0, 1], size=len(data))


import matplotlib.pyplot as plt

def calculate_trendline(data):
    x = np.array(range(len(data)))
    y = data['close'].values
    coefficients = np.polyfit(x, y, 1)
    return coefficients[0], coefficients[1], x  # Slope and intercept
 

trendlines = []
slopes = []
intercepts = []
window_size = 89
# Loop through the data with a rolling window
# for i in range(len(data) - window_size + 1):
#     window_data = data.iloc[i:i+window_size]
#     slope, intercept, x = calculate_trendline(window_data)
#     trendline = slope * x + intercept  # Line equation: y = mx + b
    
#     trendlines.append(trendline)
#     slopes.append(slope)
#     intercepts.append(intercept)
    
#     # Optionally, add the trendline to the dataframe
#     data.loc[i + window_size - 1, 'slope'] = slope
#     data.loc[i + window_size - 1, 'trendline'] = trendline[-1]



for i in range(len(data) - window_size + 1):
    # Clear the plot for each iteration
    plt.clf()
    
    window_data = data.iloc[i:i+window_size]
    slope, intercept, x = calculate_trendline(window_data)
    trendline = slope * x + intercept  # Line equation: y = mx + b
    
    # Plot the closing prices
    plt.plot(data['close'], label='Closing Prices', color='blue')
    
    # Plot the current window's trendline
    plt.plot(x + i, trendline, label=f'Trendline (Window {i+1})', linestyle='--', color='red')
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Closing Prices and Trendline (Window {i+1})')
    plt.legend()
    
    # Show the plot for this window
    plt.pause(0.1)  # Pause for 1 second to visualize each trendline (adjust time as needed)

plt.show()  


# data['inputs'] = data.apply(lambda row: [
#     row['trendline'] ,
#     row['price_diff'],     
#     row['tick_volume'],          
#     row['close'], 
#     row['slope']            
# ], axis=1)


# print(data['inputs'])

# def fitness_function(genomes, config):
#     for genome_id, genome in genomes:
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         fitness = 0
#         holding_position = False  # Initially not holding any position
#         sum_profit = 0  # Initialize sum_profit here to avoid errors

#         for _, row in data.iterrows():
#             inputs = row['inputs']
#             open_trade, close_trade = net.activate(inputs)[:2]  # Ensure we get two values (open and close)
#             if(open_trade != 0 or close_trade != 0):
#                 print(open_trade,close_trade)

#             # Threshold the outputs for trade decision
#             if open_trade > 0.5 and not holding_position:
#                 holding_position = True
#                 entry_price = row['close']  # Store entry price when opening a trade
#                 fitness -= row['price_diff']  # Cost to open the trade (could be a negative impact)
#                 print(open_trade)
#                 print(f"Trade Opened - Fitness: {fitness}, Entry Price: {entry_price}")
            
#             if close_trade > 0.5 and holding_position:
#                 holding_position = False
#                 price_diff_at_close = row['close'] - entry_price  # Profit from closing the trade
#                 fitness += row['price_diff']  # Add the profit from closing the trade
#                 sum_profit += price_diff_at_close
#                 print(open_trade)
#                 print(f"Trade Closed - Fitness: {fitness}, Profit: {price_diff_at_close} , close traded proce {row['close']}")

#             # Optional penalty for not closing a trade
#             if open_trade <= 0.5 and close_trade <= 0.5 and holding_position:
#                 fitness -= 0.1  # Small penalty for inactivity

#             # print(f"close price: {row["close"]} ") 
#         genome.fitness = fitness
        

# # NEAT config file path
# config_path = './python/config-feedforward'
# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                      config_path)

# population = neat.Population(config)
# winner = population.run(fitness_function, 10)  # Run for 100 generations

# print("Best network's fitness:", winner.fitness)











# import neat
# import numpy as np

# # Define the trading environment (simplified)
# class TradingEnvironment:
#     def __init__(self, data):
#         self.data = data  # Simulated price data
#         self.current_step = 0
#         self.cash = 1000  # Initial cash
#         self.holding = 0  # Initially, no position is held

#     def step(self, action):
#         price = self.data[self.current_step]
#         reward = 0

#         if action == 1 and self.cash >= price:  # Buy
#             self.holding = 1
#             self.cash -= price
#         elif action == 2 and self.holding:  # Sell
#             self.holding = 0
#             self.cash += price
#             reward = self.cash - 1000  # Profit or loss

#         self.current_step += 1
#         return reward

# def eval_genome(genome, config):
#     # Create the neural network for the genome
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
    
#     # Initialize environment or data (replace this with your actual environment logic)
#     env = TradingEnvironment(data=np.random.random(100))  # Mock price data

#     fitness = 0
#     for _ in range(100):  # Run the simulation for 100 steps
#         # Example input, could be slope of trend, price, etc.
#         inputs = [env.data[env.current_step], env.holding]
        
#         # Get network output (decisions)
#         output = net.activate(inputs)
        
#         # Choose action (open trade, close trade, or hold)
#         action = np.argmax(output)  # This selects the highest output
        
#         # Step the environment and get reward
#         reward = env.step(action)  # Calculate reward based on action
        
#         fitness += reward  # Accumulate profit/loss

#     return fitness

# def run_neat(config_path):
#     # Load NEAT configuration
#     config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                          config_path)

#     # Create population
#     population = neat.Population(config)

#     # Run NEAT for 50 generations
#     winner = population.run(eval_genome, 50)

#     # Print the best-performing genome
#     print("Best performing network:", winner)

# # Provide the correct path to your config file
# config_path = './python/config-feedforward'
# run_neat(config_path)