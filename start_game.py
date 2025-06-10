import json
from game.game import setup_config, start_poker
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from agents.monteoc_direct_player import setup_ai as monteoc_direct_ai
from agents.monteoc_player import setup_ai as monteoc_ai
from agents.pure_monte_player import setup_ai as pure_monte_ai
from agents.direct_player import setup_ai as direct_ai
from agents.rl_player import setup_ai as rl_ai

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="p1", algorithm=baseline5_ai())
config.register_player(name="p2", algorithm=rl_ai())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
game_result = start_poker(config, verbose=1)

print(json.dumps(game_result, indent=4))