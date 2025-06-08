from game.players import BasePokerPlayer
from agents.helper import preflop_table_lookup

class DirectPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        return preflop_table_lookup(valid_actions, hole_card, round_state, self.first_player)
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.first_player = None

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        if self.first_player is None:
            self.first_player = action["player_uuid"]

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return DirectPlayer()
