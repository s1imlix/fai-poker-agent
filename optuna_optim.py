import optuna
from functools import partial
from multiprocessing import Pool
from agents.monte_optuna_player import setup_ai as MonteCarloPlayer
from game.game import setup_config, start_poker
from baseline5 import setup_ai as baseline_ai


def play_single_game(player_class):
    """
    Run a single game using the provided player_class for p2.
    player_class should be a callable that returns a new player instance.
    """
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="p1", algorithm=baseline_ai())
    config.register_player(name="p2", algorithm=player_class())
    result = start_poker(config, verbose=0)
    # Return 1 if p2 wins, 0 otherwise
    stacks = {p["name"]: p["stack"] for p in result["players"]}
    return 1 if stacks["p2"] > stacks["p1"] else 0

def objective(trial, baseline_ai):
    player = MonteCarloPlayer()
    
    # Simulation parameters
    player.nb_simulation = trial.suggest_int('nb_simulation', 500, 5000)
    player.monte_carlo_mode = trial.suggest_categorical('monte_carlo_mode', ['fixed', 'normal'])
    
    # Overcommit ratios with ordering constraint
    player.oc_ratio_ranges = [
        trial.suggest_float('oc_ratio_low', 0.01, 0.2),
        trial.suggest_float('oc_ratio_high', 0.05, 0.3)
    ]
    trial.set_user_attr('constraint_oc_ratio', player.oc_ratio_ranges[0] < player.oc_ratio_ranges[1])
    
    # Action ratios with probability constraints
    call_ratio = trial.suggest_float('call_ratio', 0.7, 0.95)
    raise_ratio = trial.suggest_float('raise_ratio', 0.05, 0.3)
    trial.set_user_attr('constraint_action_ratio', call_ratio + raise_ratio <= 1.0)
    player.action_ratio = [0, call_ratio, raise_ratio]  # Fold ratio derived

    # Win rate thresholds with poker logic constraints
    player.raise_win_rates = [
        trial.suggest_float('sb_raise_win_rate', 0.6, 0.75),
        trial.suggest_float('bb_raise_win_rate', 0.65, 0.8)
    ]
    player.fold_win_rates = [
        trial.suggest_float('sb_fold_win_rate', 0.05, 0.2),
        trial.suggest_float('bb_fold_win_rate', 0.1, 0.25)
    ]
    
    # Overcommit thresholds with logical ordering
    player.overcommit_win_rates = [
        trial.suggest_float('sb_overcommit_win_rate', player.raise_win_rates[0]+0.05, 0.85),
        trial.suggest_float('bb_overcommit_win_rate', player.raise_win_rates[1]+0.05, 0.9)
    ]
    
    # Parallel evaluation
    with Pool() as pool:
        results = pool.map(play_single_game, [player] * 100)

    win_rate = sum(results) / len(results)
    
    # Add penalty for constraint violations
    if not trial.user_attrs['constraint_oc_ratio']:
        win_rate -= 0.2
    if not trial.user_attrs['constraint_action_ratio']:
        win_rate -= 0.2
        
    return win_rate



def create_optimize_study(baseline_ai):
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///monte_optuna_study.db',
        study_name='monte_optuna_study',
    )
    study.optimize(partial(objective, baseline_ai=baseline_ai), n_trials=200)
    return study

def main():
    study = create_optimize_study(baseline_ai)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  Constraints: {trial.user_attrs}")

if __name__ == "__main__":
    main()
