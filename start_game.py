import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from baseline0 import setup_ai as baseline0_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from agents.monte_player import setup_ai as monte_ai

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="p1", algorithm=baseline5_ai())
config.register_player(name="p2", algorithm=monte_ai())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
game_result = start_poker(config, verbose=1)

print(json.dumps(game_result, indent=4))