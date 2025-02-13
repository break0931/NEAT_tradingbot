import neat
import pandas as pd
import numpy as np


data = pd.read_csv("XAUUSD_16408_6years_89.csv")

print(data)



# Simulated environment
class TradingEnvironment:
    def __init__(self, data, capital=10000, leverage=1, spread_rate=0.0):
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

        else:  # action == 0 ถือเฉยๆ
            reward -= (self.capital * 0.001)  # ค่าปรับสำหรับการไม่ทำอะไรเลย
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
    with open("best_TRAIN01_D1_close-trendline_5yearfile.pkl", "rb") as f:
        import pickle
        best_genome = pickle.load(f)

    # Load unseen test data
    test_data = data[383:748]  # Last 20% of the dataset
    test_data['close_minus_trendline'] = test_data['close'] - test_data['trendline']
    print(test_data)

    # Test the best genome
    test_best_genome(best_genome, config, test_data)





