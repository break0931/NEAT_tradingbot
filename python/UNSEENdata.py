import neat
import pandas as pd
import numpy as np
import os

file_path = "./best_genome.pkl"
if not os.path.exists(file_path):
    print("File does not exist.")
elif os.path.getsize(file_path) == 0:
    print("File is empty.")
else:
    print("File exists and is not empty.")


data = pd.read_csv("updated_output.csv")
test_data = data[int(0.8 * len(data)):]
feature_columns = [  'trendline' ,'price_diff', 'tick_volume',  'close', 'time_sin' , 'time_cos' ]
test_inputs = test_data[feature_columns].values
import neat

# Load the NEAT configuration
config_path = "neat-config.txt"  # Path to your NEAT configuration file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

import pickle

with open("./best_genome.pkl", "rb") as f:
    winner_genome = pickle.load(f)


model = neat.nn.FeedForwardNetwork.create(winner_genome, config)


# Define the environment simulation
class TestEnvironment:
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.holding = 0  # Initial holding state
    
    def get_state(self):
        if self.index >= len(self.data):
            return None  # End of data
        
        row = self.data.iloc[self.index]
        trendline = row['trendline']
        price_diff = row['price_diff']
        volume = row['tick_volume']
        close = row['close']
        time_sin = row['time_sin']
        time_cos = row['time_cos']
        holding = self.holding  # Use current holding state
        
        self.index += 1  # Move to the next data point
        return [trendline, price_diff, volume, close, time_sin, time_cos, holding]

# Initialize the environment
env = TestEnvironment(test_data)

# Test the model on unseen data
predictions = []
states = []

while True:
    state = env.get_state()
    if state is None:
        break  # End of data
    
    prediction = model.activate(state)  # Get model prediction
    predictions.append(prediction)
    states.append(state)

# Combine predictions with test data for evaluation
test_data['predictions'] = predictions

# Display or save results
print(test_data[['trendline', 'price_diff', 'tick_volume', 'close', 'predictions']])