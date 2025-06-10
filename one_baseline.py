import json
from game.game import setup_config, start_poker
from agents.monteoc_player import setup_ai as monteoc_ai
from agents.monteoc_direct_player import setup_ai as monteoc_direct_ai
from agents.direct_player import setup_ai as direct_ai
from agents.rl_player import setup_ai as rl_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline4 import setup_ai as baseline4_ai
from baseline3 import setup_ai as baseline3_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from multiprocessing import Pool

def create_config():
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=baseline5_ai())
    config.register_player(name="p2", algorithm=rl_ai())
    return config

def play_game(_):
    config = create_config()
    result = start_poker(config, verbose=0)
    seats = {p["name"]: p["stack"] for p in result["players"]}
    winner = max(seats, key=seats.get)
    return winner

def run_parallel_games(n_runs=100):
    with Pool() as pool:
        winners = pool.map(play_game, range(n_runs))

    p2_wins = sum(1 for winner in winners if winner == "p2")
    print(f"Player 2: {p2_wins}/{n_runs} wins ({(p2_wins / n_runs) * 100:.2f}%)")


if __name__ == "__main__":
    run_parallel_games(100)  # Adjust the number of runs as needed