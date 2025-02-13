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
data = pd.read_csv("XAUUSD_2018-08-04_00-00-00_2025-02-11_00-00-00_16408.csv")
data['close_minus_trendline'] = data['close'] - data['trendline']
# Define the trading environment
class TradingEnvironment:
    def __init__(self, data, capital=10000, leverage=1, spread_rate=0.0005):
        self.data = data
        self.capital = capital
        self.leverage = leverage
        self.spread_rate = spread_rate
        self.position = 0  # จำนวนสถานะที่เปิดอยู่ (สามารถเป็น +5 ถึง -5)
        self.entry_prices = []  # เก็บราคาเปิดของแต่ละสถานะ
        self.index = 0  # ตำแหน่งของข้อมูลที่ใช้
        
    def get_state(self):
        if self.index >= len(self.data):
            return None
        trendline = self.data.iloc[self.index]['trendline']
        price_diff = self.data.iloc[self.index]['price_diff']
        volume = self.data.iloc[self.index]['tick_volume']
        close = self.data.iloc[self.index]['close']
        time_sin = self.data.iloc[self.index]['time_sin']
        time_cos = self.data.iloc[self.index]['time_cos']
        close_minus_trendline = self.data.iloc[self.index]['close_minus_trendline']
        return [close_minus_trendline, price_diff, volume, self.position ]
    
    def step(self, action):
        current_price = self.data.iloc[self.index]['close']
        reward=0
        profit =0
        if action == 1:  # เปิด Long
            if self.position < 0:  # มี Short อยู่ ต้องปิดก่อน
                profit = self.close_all_positions(current_price)

            if self.position < 5:  # เปิดเพิ่มได้สูงสุด 5 ตำแหน่ง
                self.position += 1
                self.entry_prices.append(current_price)
           
        elif action == 2:  # เปิด Short
            if self.position > 0:  # มี Long อยู่ ต้องปิดก่อน
                profit = self.close_all_positions(current_price)
            if self.position > -5:  # เปิดเพิ่มได้สูงสุด -5 ตำแหน่ง
                self.position -= 1
                self.entry_prices.append(current_price)
        # else:  # action == 0 ถือเฉยๆ
        #     reward -= (self.capital * 0.001)  # ค่าปรับสำหรับการไม่ทำอะไรเลย
        
        
        self.index += 1  # ขยับไปยังข้อมูลถัดไป
        reward += profit

        return self.get_state(), reward ,profit

    def close_all_positions(self, current_price):
        """ ปิดสถานะทั้งหมด และคำนวณกำไร/ขาดทุน """
        raw_profit = 0
        for entry_price in self.entry_prices:
            trade_profit = (current_price - entry_price) if self.position > 0 else (entry_price - current_price)
            raw_profit += trade_profit * self.leverage * ((self.capital / 5) / entry_price) 

        spread_cost = abs(raw_profit) * self.spread_rate
        profit = raw_profit - spread_cost
        self.capital += profit
        
       
        # รีเซ็ตสถานะ
        self.position = 0
        self.entry_prices = []
        return profit


# Fitness evaluation function
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    train_data = data[90:]
    env = TradingEnvironment(train_data)
    total_profit = 0
    total_reward = 0
    actions_log = []
    long_trades = 0
    short_trades = 0
    for _ in range(len(train_data)-10):
        state = env.get_state()
        if state is None:
            break
        
        
        action = np.argmax(net.activate(state))
        _, reward,profit = env.step(action)

        total_reward += reward
        total_profit += profit
       
        if action == 1:  # Long trades
            long_trades += 1
        elif action == 2:  # Short trades
            short_trades += 1
        actions_log.append({
            "index": env.index - 1 + 1,
            "state": state,
            "action": action,
            "reward": reward,
            "capital": env.capital,
            'close': env.data.iloc[env.index]['close']
        })
    # Penalize imbalanced behavior1
    #balance_penalty = -(abs(long_trades - short_trades) / (len(train_data))) * total_profit * 0.3
    genome.fitness = total_reward #+ balance_penalty

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
    reward, profit, actions_log = evaluate_genome(winner, config)
    for log_entry in actions_log:
        log_line = (
            f"Step {log_entry['index']}:\n"
            f"  State: {log_entry['state']}\n"
            f"  Action: {log_entry['action']}\n"
            f"Capital: {log_entry['capital']:.2f}\n"
            f"Close: {log_entry['close']:.2f}\n"
        )
        print(log_line)
    bestgenome_action_counts = {"long": 0, "short": 0, "hold": 0}
     # Count actions from the genome's logs
    for log_entry in actions_log:
        action = log_entry["action"]
        if action == 1 :  # Long actions (1 = open, 2 = close)
            bestgenome_action_counts["long"] += 1
        elif action == 2:  # Short actions (3 = open, 4 = close)
            bestgenome_action_counts["short"] += 1
        elif action == 0:  # Hold action
            bestgenome_action_counts["hold"] += 1
    
    print(f"\nBestgenome Action Counts:\n{bestgenome_action_counts}\n")

    # Save the best genome
    with open("best_TRAIN04_D1_close-trendline_89.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:\n", winner)
    print("\nProfit:\n", profit)
    print("\nFitness:\n", reward)


# Fitness function wrapper for NEAT
def fitness_function(genomes, config):
    global_action_counts = {"long": 0, "short": 0, "hold": 0}

    for genome_id, genome in genomes:
        total_reward, total_profit, actions_log = evaluate_genome(genome, config)

        # Count actions from the genome's logs
        for log_entry in actions_log:
            action = log_entry["action"]
            if action == 1:  
                global_action_counts["long"] += 1
            elif action == 2:  
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
