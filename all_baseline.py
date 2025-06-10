import json
from game.game import setup_config, start_poker
from agents.monteoc_direct_player import setup_ai as monteoc_direct_ai
from agents.monteoc_player import setup_ai as monteoc_ai
from agents.direct_player import setup_ai as direct_ai
from agents.pure_monte_player import setup_ai as pure_monte_ai
from agents.rl_player import setup_ai as rl_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from multiprocessing import Pool

all_ais = [baseline1_ai, baseline2_ai, baseline3_ai, baseline4_ai, baseline5_ai, baseline6_ai, baseline7_ai]
output = []
def create_config(index):
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=all_ais[index]())
    config.register_player(name="p2", algorithm=monteoc_ai())
    return config

def play_game(index):
    config = create_config(index)
    result = start_poker(config, verbose=0)
    seats = {p["name"]: p["stack"] for p in result["players"]}
    winner = max(seats, key=seats.get)
    return winner

def run_parallel_games_for_ai(ai_index, n_runs=100):
    global output
    with Pool() as pool:
        winners = pool.map(play_game, [ai_index] * n_runs)
    p2_wins = sum(1 for winner in winners if winner == "p2")
    output.append(f"AI {ai_index+1}: Player 2: {p2_wins}/{n_runs} wins ({(p2_wins / n_runs) * 100:.2f}%)")

def run_all_ais(n_runs=100):
    global output
    for i, ai_class in enumerate(all_ais):
        run_parallel_games_for_ai(i, n_runs)
    
    print("\n".join(output))

if __name__ == "__main__":
    run_all_ais(100)  # Each AI plays 100 games
