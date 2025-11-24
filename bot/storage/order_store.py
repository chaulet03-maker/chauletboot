import json
import os

class OrderStore:
    PATH = "paper_positions.json"

    def __init__(self):
        if not os.path.exists(self.PATH):
            with open(self.PATH, "w") as f:
                json.dump([], f)

    def load_positions(self):
        try:
            with open(self.PATH, "r") as f:
                return json.load(f)
        except:
            return []

    def save_position(self, pos: dict):
        positions = self.load_positions()
        positions.append(pos)
        with open(self.PATH, "w") as f:
            json.dump(positions, f, indent=2)

    def clear(self):
        with open(self.PATH, "w") as f:
            json.dump([], f)
