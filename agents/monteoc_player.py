from game.players import BasePokerPlayer
from agents.hand_evaluator import HandEvaluator
from agents.card_utils import estimate_hole_card_win_rate, gen_cards, evaluate_hand
from agents.helper import preflop_table_lookup
import random as rand
import numpy as np

class MonteCarloPlayer(
    BasePokerPlayer
): 
    def __init__(self):
        self.nb_simulation = 1000

        self.oc_ratio_range = [0.05, 0.1]

        self.monte_carlo_mode = "fixed"  
        # "normal": normal distribution with mean on threshold
        # "fixed": fixed threshold
        self.normal_std = 0.02  # Standard deviation for normal distribution

        self.progress_adjust = False
        self.progress_adjust_window = 0.05
        

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
    def is_SB(self) -> bool:
        return self.first_player is None or self.first_player == self.uuid

    @property
    def action_ratio(self) -> tuple: # fold, call, raise
        if self.is_SB:
            return [0, 0.95, 0.05] # first-hand, raise more
        else:
            return [0, 0.95, 0.05]  

    @property
    def overcommit_ratio(self) -> float:
        if self.is_behind:
            return self.oc_ratio_range[0] + self.progress * (self.oc_ratio_range[1] - self.oc_ratio_range[0]) 
        else:
            return self.oc_ratio_range[1] - self.progress * (self.oc_ratio_range[1] - self.oc_ratio_range[0])

    @property
    def no_give_up_ratio(self) -> float:
        return self.overcommit_ratio + 0.05

    @property
    def max_raise_ratio(self) -> float:
        return self.overcommit_ratio 

    @property
    def oppo_overcommit_ratio(self) -> float:
        return self.overcommit_ratio 

    @property
    def raise_win_rate(self) -> float:
        if self.is_SB:
            p = 0.65
        else:
            p = 0.7 
        return p + self.progress_adjust_window * self.progress if self.progress_adjust else p

    
    @property
    def overcommit_win_rate(self) -> float:
        return self.raise_win_rate + 0.1

    @property
    def overcommit_recheck_win_rate(self) -> float:
        if self.is_behind:
            p = 0.55 
        else:
            p = 0.6
        return p - self.progress_adjust_window * self.progress if self.progress_adjust else p


    @property
    def fold_win_rate(self) -> tuple:  
        if self.is_behind:
            return 0.1
        return 0.15

    @property
    def bet_ratio(self) -> float:
        return (self.total_bet + self.street_bet) / self.round_start_stack
    
    @property
    def oppo_bet_ratio(self) -> float:
        return (self.oppo_total_bet + self.oppo_street_bet) / self.oppo_round_start_stack

    @property
    def is_safe_fold(self) -> bool:
        return self.bet_ratio < self.no_give_up_ratio and ((not self.is_behind and self.total_bet - self.diff < 5) or # winning and this bet doesn't exceed our lead, we allow 1-round
                not self.safe_round_check(self.oppo_round_start_stack + self.total_bet, self.remaining_rounds - 1)) # losing but no safe round

    def need_recheck(self, win_rate) -> bool:
        return (
                (win_rate < self.overcommit_win_rate) and # not high enough
                ((win_rate > self.overcommit_recheck_win_rate and self.oppo_bet_ratio > self.oppo_overcommit_ratio) or
                 (win_rate < self.overcommit_recheck_win_rate and win_rate > self.fold_win_rate)
                ) # aggressive opponent or pretty low win rate
            )   

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
        if self.monte_carlo_mode == "normal":
            win_rate_samples = np.random.normal(loc=win_rate, scale=self.normal_std, size=100)
            raise_prob = sum(win_rate_samples >= self.raise_win_rate) / len(win_rate_samples)
            fold_prob = sum(win_rate_samples <= self.fold_win_rate) / len(win_rate_samples)
            call_prob = 1 - raise_prob - fold_prob
            print(f">>> Sampled weights: raise {raise_prob:.2f}, fold {fold_prob:.2f}, call {call_prob:.2f}")
            choice = self.choose_with_ratio(["raise", "fold", "call"], [raise_prob, fold_prob, call_prob])
            choice = self.choose_action(valid_actions, choice)
        elif self.monte_carlo_mode == "fixed":
            if (win_rate >= self.raise_win_rate):
                choice = self.choose_action(valid_actions, "raise")
            elif (win_rate <= self.fold_win_rate):
                choice = self.choose_action(valid_actions, "fold")
            else:
                choice = self.choose_with_ratio(["call", "raise"], self.action_ratio[1:])
                choice = self.choose_action(valid_actions, choice)

        
        action = choice["action"]
        amount = choice["amount"]

        if action == "raise":
            if amount["min"] == -1:
                action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
            else:
                self.have_raised = True
                max_allowed = max(self.max_raise_ratio * amount["max"], amount["min"])
                # Exponential scaling curve
                exponent = 10 
                scaled_win_rate = win_rate ** exponent 
                amount = amount["min"] + scaled_win_rate * (max_allowed - amount["min"])

        '''
        if round_state["street"] == "preflop":
            # Preflop logic, use direct player for preflop decision
            action, amount = preflop_table_lookup(valid_actions, hole_card, round_state, self.first_player)
        '''
        
        print(f">>> Action chosen: {action}, amount: {amount}")

        new_bet = self.total_bet + amount
        new_bet_ratio = (new_bet / self.round_start_stack)
        print(f">>> Over-commit check: new_bet_ratio = {new_bet_ratio:.2f}, bet_ratio = {self.bet_ratio:.2f}, oppo_bet_ratio = {self.oppo_bet_ratio:.2f}")
        """
        Over-commit check
        """
        if (
            self.remaining_rounds > 1 and
            new_bet_ratio > self.overcommit_ratio 
        ): 
            print(f"|--- Over-commit detected: new_bet_ratio {new_bet_ratio:.2f} > overcommit_ratio {self.overcommit_ratio:.2f}")
            if (self.need_recheck(win_rate)):
                print(f"|--- Win-rate {win_rate:.2f} < overcommit_win_rate {self.overcommit_win_rate:.2f} or aggresive opponent, re-check")
                if (action == "raise"): 
                    if (self.is_safe_fold):
                        print(f"    |--- win_rate {win_rate:.2f} < call_win_rate and will cause oppo safe round, or we're not losing (bet: {self.total_bet:.2f} < diff: {self.diff:.2f}), fold")
                        return valid_actions[0]["action"], valid_actions[0]["amount"]
                    else:
                        return valid_actions[1]["action"], valid_actions[1]["amount"]
                elif (self.oppo_bet_ratio > self.oppo_overcommit_ratio and self.is_safe_fold):
                    print(f"    |--- calling on aggressive opponent, fold")
                    return valid_actions[0]["action"], valid_actions[0]["amount"]

        """
        Edge cases
        """

        print(f">>> Final action: {action}, amount: {amount}")
        expected_val = (self.oppo_total_bet + self.oppo_street_bet) * win_rate - (self.total_bet + self.street_bet) * (1 - win_rate) if action != "fold" else - (self.total_bet + self.street_bet)
        print(f">>> Expected income: {expected_val:.2f} (oppo_bet: {self.oppo_total_bet + self.oppo_street_bet:.2f}, win_rate: {win_rate:.2f})")

        self.street_bet = amount
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
        self.progress = self.remaining_rounds / self.total_rounds
        self.safe_round = self.safe_round_check(self.round_start_stack, self.remaining_rounds)
        self.diff = self.round_start_stack - self.oppo_round_start_stack
        self.is_behind = self.diff < 0
        self.have_raised = False

        self.total_bet = 7.5
        self.oppo_total_bet = 7.5
        self.street_bet = 0
        self.oppo_street_bet = 0

        self.first_player = None

    def receive_street_start_message(self, street, round_state):
        self.total_bet += self.street_bet 
        self.oppo_total_bet += self.oppo_street_bet
        self.street_bet = 0
        self.oppo_street_bet = 0

    def receive_game_update_message(self, action, round_state):
        if self.first_player is None:
            self.first_player = action["player_uuid"]
        if action["player_uuid"] != self.uuid and (action["action"] == "raise" or action["action"] == "call"):
            self.oppo_street_bet = action["amount"]

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return MonteCarloPlayer()