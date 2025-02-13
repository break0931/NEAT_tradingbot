import zmq
import json
import pandas as pd
import numpy as np
import neat
import pickle

def main():
    np.set_printoptions(suppress=True, precision=6)

    with open('best_TRAIN04_D1_close-trendline_89.pkl', 'rb') as f:
        winner_genome = pickle.load(f)

    config_path = "neat-config.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")

    print("Python Server: Waiting for messages...")
    socket_send = context.socket(zmq.REQ)
    socket_send.connect("tcp://127.0.0.1:5566")
    # Initialize DataFrame
    df = pd.DataFrame(columns=["Close","close_minus_trendline", "Volume", "Position", "trendline","price_diff"]).astype({
        "Close": "float",
        "close_minus_trendline": "float",
        "Volume": "int",
        "Position": "int",
        "trendline": "float",
        "price_diff": "float"
    })

    window_size = 89

    def calculate_trendline(data):
        x = np.array(range(len(data)))
        y = data['Close'].values
        coefficients = np.polyfit(x, y, 1)
        return coefficients[0], coefficients[1], x

    while True:
        message = socket.recv()
        message_str = message.decode()
        print(message_str)
        try:
            # Load JSON data
            data_list = json.loads(message_str)
            new_rows = pd.DataFrame(data_list)

            # Ensure numeric types
            new_rows = new_rows.astype({
                "Close": "float",
                "Volume": "int",
                "Position": "int"
            })

            # Drop datetime column if it exists
            if 'datetime' in new_rows.columns:
                new_rows.drop(columns=['datetime'], inplace=True)

            # Append new data to df and keep only the last 89 rows
            df = pd.concat([df, new_rows], ignore_index=True).tail(89)

            # Calculate `price_diff`
           

            # Calculate trendline if we have enough data
            if len(df) >= window_size:
                slope, intercept, x = calculate_trendline(df.iloc[-window_size:])  # ใช้ 89 แท่งล่าสุด
                df.loc[df.index[-1], 'trendline'] = slope * x[-1] + intercept  # ค่าของแท่งสุดท้ายในหน้าต่าง

            # for i in range(len(df) - window_size + 1):
            
            #     window_data = df[i:i+window_size]
            #     slope, intercept, x = calculate_trendline(window_data)
            #     trendline = slope * x + intercept  # Line equation: y = mx + b
            #     df[i + window_size - 1, 'trendline'] = trendline[-1]
            df["price_diff"] = df["Close"].diff()
            df["close_minus_trendline"] = df["Close"] - df["trendline"]


            # Fill NaN values
            df.fillna(0, inplace=True)

            #print(df.tail(89))  # Show last 10 rows

        except json.JSONDecodeError:
            print("Error: Invalid JSON format")
        except KeyError as e:
            print(f"KeyError: {e}")

        # Get the latest row
        latest_data = df.tail(1)[["close_minus_trendline","price_diff", "Volume", "Position"]].values.flatten()
        latest_data = latest_data.astype(float)
        print(latest_data)

        raw_output = winner_net.activate(latest_data)   
        print(raw_output) 
        # Make prediction
        action = np.argmax(winner_net.activate(latest_data))
        # epsilon = 0.1
        # if np.random.rand() < epsilon:
        #     action = np.random.choice(len(winner_net.activate( latest_data)))  # Random exploration
        # else:
        #     action = np.argmax(winner_net.activate( latest_data))

        print("prediction",action)
        socket.send_string(str(action))



        
      
        #socket_send.send_string(str(action))
if __name__ == "__main__":
    main()
