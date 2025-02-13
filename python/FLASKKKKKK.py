from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/trade', methods=['POST'])
def trade():
    data = request.json
    symbol = data.get("symbol")
    volume = data.get("volume")
    return jsonify({"status": "success", "message": f"Received trade for {symbol} with volume {volume}"})

if __name__ == '__main__':
    app.run(debug=True)
