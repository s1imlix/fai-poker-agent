import json
from game.game import setup_config, start_poker

from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from agents.monte_player import setup_ai as player_ai

from multiprocessing import Pool, cpu_count

def play_match(p1_algo, p2_algo, rounds=20):
    config = setup_config(
        max_round=rounds,
        initial_stack=1000,
        small_blind_amount=5
    )
    config.register_player(name="p1", algorithm=p1_algo)
    config.register_player(name="p2", algorithm=p2_algo)
    result = start_poker(config, verbose=0)
    seats = {p["name"]: p["stack"] for p in result["players"]}
    return seats["p1"], seats["p2"]

def score_and_detail(p1_factory, p2_factory):
    matches = []
    p2_win_count = 0

    for i in range(5):
        p1_s, p2_s = play_match(p1_factory(), p2_factory())
        did_win = (p2_s > p1_s)
        matches.append({
            "p1_stack": p1_s,
            "p2_stack": p2_s,
            "p2_win": did_win
        })
        if did_win:
            p2_win_count += 1

    if p2_win_count >= 3:
        score = 5.0
    else:
        sorted_matches = sorted(matches, key=lambda m: m["p2_stack"], reverse=True)
        score = 0.0
        for m in sorted_matches[:2]:
            if m["p2_win"]:
                score += 1.5
            else:
                if m["p2_stack"] >= 500:
                    score += round((m["p2_stack"] / 1000.0), 1)
    return matches, score

def worker_baseline(args):
    name, p1_factory = args
    matches, score = score_and_detail(p1_factory, player_ai)
    return name, matches, score

def main():
    baselines = [
        ("baseline1", baseline1_ai),
        ("baseline2", baseline2_ai),
        ("baseline3", baseline3_ai),
        ("baseline4", baseline4_ai),
        ("baseline5", baseline5_ai),
        ("baseline6", baseline6_ai),
        ("baseline7", baseline7_ai),
    ]

    num_processes = 7

    with Pool(processes=num_processes) as pool:
        results_list = pool.map(worker_baseline, baselines)

    results = {}
    total_score = 0.0
    for name, matches, score in results_list:
        total_score += score
        results[name] = {
            "matches": matches,
            "score": score
        }

    results["total_score"] = f"{total_score:.3f}/35.000"

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
