from game.players import BasePokerPlayer
from agents.hand_evaluator import HandEvaluator
from agents.card_utils import estimate_hole_card_win_rate, gen_cards, evaluate_hand
from agents.helper import sigmoid
import random as rand

class MonteCarloPlayer(
    BasePokerPlayer
): 
    def __init__(self):
        self.action_ratio = [0, 0.95, 0.05]  # fold, call, raise
        self.nb_simulation = 1000

    def safe_round_check(self, balance, remain):
        return (balance - 1000) > 7.5 * remain

    def choose_action(self, valid_actions, action_type):
        for choice in valid_actions:
            if choice["action"] == action_type:
                return choice 
        return valid_actions[0]  # Default to fold if no valid action found

    def choose_with_ratio(self, actions, ratios):
        return rand.choices(
            actions,
            weights=ratios,
            k=1
        )[0]  # Return a single action based on the given ratios
    
    @property
    def raise_win_rate(self) -> float:
        if self.is_behind:
            return 0.65
        return 0.7
    
    @property
    def fold_win_rate(self) -> tuple:  
        if self.is_behind:
            return 0.1
        return 0.15
    

    def declare_action(self, valid_actions, hole_card, round_state):

        if self.safe_round:
            print(">>> Safe round detected, folding.")
            return valid_actions[0]["action"], valid_actions[0]["amount"]

        hole_cards = gen_cards(hole_card)
        community_cards = gen_cards(round_state["community_card"])

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=self.nb_simulation,
            hole_card=hole_cards,
            community_card=community_cards,
            nb_player=2
        )

        print(f">>> Win rate: {win_rate:.2f}")

        """
        Monte Carlo decision making
        """

        if (win_rate >= self.raise_win_rate):
            choice = self.choose_action(valid_actions, "raise")
        elif (win_rate <= self.fold_win_rate):
            choice = self.choose_action(valid_actions, "fold")
        else:
            choice = self.choose_with_ratio(["call", "raise"], self.action_ratio[1:])
            choice = self.choose_action(valid_actions, choice)
        
        action = choice["action"]
        amount = choice["amount"]

        if action == "raise": # determine raise amount
            if (amount["min"] == -1):
                action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
            else: 
                self.have_raised = True   
                action, amount = choice["action"], amount["min"]

        return action, amount
    

    def receive_game_start_message(self, game_info):
        self.total_rounds = game_info["rule"]["max_round"]


    def receive_round_start_message(self, round_count, hole_card, seats):
        
        """
        First check if we can just give up, each round folding at most lose 10 chips, 
        so if our - theirs > 10 * (total_rounds - round_count), we can just fold till the end
        """
        for player in seats:
            if player["uuid"] == self.uuid:
                self.round_start_stack = player["stack"]
            else:
                self.oppo_round_start_stack = player["stack"]

        # we alternate between small blind and big blind, so we can use 7.5 as average fold loss
        self.remaining_rounds = self.total_rounds - round_count + 1
        self.safe_round = self.safe_round_check(self.round_start_stack, self.remaining_rounds)
        self.diff = self.round_start_stack - self.oppo_round_start_stack
        self.is_behind = self.diff < 0
        self.have_raised = False

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return MonteCarloPlayer()