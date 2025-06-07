from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from agents.card_utils import estimate_hole_card_win_rate, gen_cards, evaluate_hand
from agents.helper import sigmoid
import random as rand

class MonteCarloPlayer(
    BasePokerPlayer
): 
    def __init__(self):
        self.action_ratio = [0, 0.95, 0.05]  # fold, call, raise
        self.nb_simulation = 20000


        self.raise_ratio_aimd_range = [0.01, 0.02]
        self.raise_ratio_fixed = 0.2
        self.cur_raise_ratio = 0.02  # Default raise ratio, AIMD
        self.ai_step = 0.01
        self.md_factor = 0.5

        self.raise_threshold = 0.04 # our threshold, exceeding this means we turn non-aggressive
        self.no_give_up_threshold = 0.15
        self.oppo_raise_threshold = 0.05  # Opponent's raise threshold, exceeding this means they are aggressive

    def update_raise_ratio(self, is_winner):
        if is_winner:
            self.cur_raise_ratio = min(self.raise_ratio_aimd_range[1], self.cur_raise_ratio + self.ai_step)
        else:
            self.cur_raise_ratio = max(self.raise_ratio_aimd_range[0], self.cur_raise_ratio * self.md_factor)

    def calc_raise_amount(self, available_range_dict, strat="AIMD", ratio=0.5):
        available_range = [available_range_dict["min"], available_range_dict["max"]]
        if strat == "AIMD":
            raise_amount = int(available_range[0] + self.cur_raise_ratio * (available_range[1] - available_range[0]))
        elif strat == "Rand":
            raise_amount = rand.randint(available_range[0], available_range[1])
        elif strat == "Fixed":
            raise_amount = int(available_range[0] + ratio * (available_range[1] - available_range[0]))
        return raise_amount

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        sum_ratio = fold_ratio + call_ratio + raise_ratio
        self.action_ratio = [fold_ratio / sum_ratio, call_ratio / sum_ratio, raise_ratio / sum_ratio]

    @property
    def raise_ratio(self) -> float:
        return 1.0 * (self.total_bet + self.street_bet) / self.round_start_stack
    
    @property
    def oppo_raise_ratio(self) -> float:
        return 1.0 * (self.opponent_raise + self.opponent_street_raise) / self.oppo_round_start_stack

    @property
    def must_raise_rate(self) -> float:
        return 0.85

    @property
    def raise_win_rate(self) -> float:
        return 0.75
    
    @property 
    def fold_win_rate(self) -> tuple:
        if self.is_behind:
            return 0.1, 0.3 
        return 0.15, 0.45
         
    @property 
    def overcommit_call_win_rate(self) -> float:
        return 0.68

    @property
    def overcommit_raise_win_rate(self) -> float:
        return self.raise_win_rate + 0.05

    def raise_condition(self, must_raise, win_rate) -> bool:
        return (win_rate > self.raise_win_rate and 
                self.oppo_raise_ratio < self.oppo_raise_threshold) or must_raise

    def fold_condition(self, round_state, win_rate) -> bool:
        return (round_state["street"] != "preflop" and
                (
                    (win_rate <= self.fold_win_rate[0] and self.oppo_raise_ratio > 0) or
                    (win_rate <= self.fold_win_rate[1] and (self.total_bet < self.diff * 0.5
                                                        or self.oppo_raise_ratio > self.oppo_raise_threshold)) or
                    (win_rate <= self.must_raise_rate and self.oppo_raise_ratio > self.oppo_raise_threshold * 5)
                )
               )

    def safe_round_check(self, balance, remain):
        return (balance - 1000) > 7.5 * remain

    def declare_action(self, valid_actions, hole_card, round_state):

        if self.safe_round:
            print("* Safe round detected, folding.")
            return valid_actions[0]["action"], valid_actions[0]["amount"]

        hole_cards = gen_cards(hole_card)
        community_cards = gen_cards(round_state["community_card"])

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=self.nb_simulation,
            hole_card=hole_cards,
            community_card=community_cards,
            nb_player=2
        )

        print(f">>> Win rate: {win_rate:.2f}, Raise ratio: {self.raise_ratio:.2f}, Opponent raise ratio: {self.oppo_raise_ratio:.2f}")

        must_raise = False
        if round_state["street"] == "river":
            eval_ret = evaluate_hand(hole_cards, community_cards)
            print(f">>> Evaluated hand: {eval_ret['hand']}, Strength: {eval_ret['strength']}")
            if eval_ret["hand"] != "HIGHCARD" and win_rate > self.must_raise_rate:
                must_raise = True

        """
        Monte Carlo decision making
        """

        if (self.raise_condition(must_raise, win_rate)):
            choice = self.choose_action(valid_actions, "raise")
        elif (self.fold_condition(round_state, win_rate)):
            print(f">>> Aggressive opponent: {self.oppo_raise_ratio > self.oppo_raise_threshold}")
            choice = self.choose_action(valid_actions, "fold")
        else:
            action_type = self.choose_with_ratio(["fold", "call", "raise"], self.action_ratio)
            choice = self.choose_action(valid_actions, action_type)
        
        action = choice["action"]
        amount = choice["amount"]

        if action == "raise":
            if (amount["min"] == -1):
                action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
            else:    
                if must_raise: # all in with our best!!
                    amount = self.calc_raise_amount(amount, "Fixed", self.raise_ratio_fixed)
                else:  # we can raise, but not too much
                    amount = self.calc_raise_amount(amount, "AIMD", self.raise_ratio)
                self.have_raised = True

        new_bet = self.total_bet + amount
        commit_to_full_ratio = self.raise_ratio / self.raise_threshold
        new_commit_to_full_ratio = (new_bet / self.round_start_stack) / self.raise_threshold
        print(f">>> Chosen action: {action}, Amount: {amount}, street bet: {self.street_bet}, total bet: {self.total_bet}")

        """
        Over-commit detection
        """
        if (
            not must_raise and
            self.remaining_rounds > 1 and
            new_commit_to_full_ratio > 1 and # we will be raising too much
            self.raise_ratio < self.no_give_up_threshold # but not to the point we can't give up
            ):
            print(f">>> Over-commit detected, new ratio: {new_commit_to_full_ratio:.3f}")
            
            if ((action == "raise" and win_rate < self.overcommit_raise_win_rate) or 
                  (action == "call" and win_rate < self.overcommit_call_win_rate) 
                 ): 
                print(f"|--- Win-rate {win_rate:.2f} needs re-evaluation")
                if (win_rate < self.overcommit_call_win_rate):
                    if (not self.is_behind or
                        self.safe_round_check(self.oppo_round_start_stack + new_bet, self.remaining_rounds-1)):
                        print(f"    |--- win_rate {win_rate:.2f} < call_win_rate and will cause oppo safe round, or we're not losing, fold")
                        return valid_actions[0]["action"], valid_actions[0]["amount"]
                elif commit_to_full_ratio < 1: 
                    call_prob = sigmoid(commit_to_full_ratio, sharpness=20, midpoint=1-win_rate)
                    print(f"    |--- current {100*commit_to_full_ratio:.2f}% -> Call probability: {call_prob:.2f}")
                    new_action = self.choose_action(valid_actions, 
                                                self.choose_with_ratio(["fold", "call"], [1 - call_prob, call_prob]))
                    action, amount = new_action["action"], new_action["amount"]
            

        self.street_bet = amount
        print(f">>> Final action: {action}, Amount: {amount}, street bet: {self.street_bet}, total bet: {self.total_bet}")
        return action, amount

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
    

    def receive_game_start_message(self, game_info):
        self.total_rounds = game_info["rule"]["max_round"]
        self.total_bet = 0
        self.opponent_raise = 0
        self.street_bet = 0
        self.opponent_street_raise = 0

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
        self.total_bet = 7.5
        self.opponent_raise = 7.5 
        self.street_bet = 0
        self.opponent_street_raise = 0

    def receive_street_start_message(self, street, round_state):
        self.total_bet += self.street_bet 
        self.opponent_raise += self.opponent_street_raise 
        self.street_bet = 0
        self.opponent_street_raise = 0

    def receive_game_update_message(self, action, round_state):
        if action["player_uuid"] != self.uuid and (action["action"] == "raise" or action["action"] == "call"):
            self.opponent_street_raise = action["amount"]

    def receive_round_result_message(self, winners, hand_info, round_state):
        # see if winner contains self UUID, it is winners.uuid
        is_winner = any(winner["uuid"] == self.uuid for winner in winners)
        self.update_raise_ratio(is_winner)


def setup_ai():
    return MonteCarloPlayer()
