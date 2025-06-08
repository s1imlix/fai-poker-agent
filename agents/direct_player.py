from game.players import BasePokerPlayer


class DirectPlayer(BasePokerPlayer):

    @property
    def open_range(self):
        return {
            "22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA",
            "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
            "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s",
            "AKo", "AQo", "AJo", "KQo", "KJo", "QJo", "A9o", "A8o", "A7o", "A6o", "A5o"
        }

    @property
    def call_range(self):
        return {
            "22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA",
            "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
            "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s",
            "AKo", "AQo", "AJo", "KQo", "KJo", "QJo", "A9o", "A8o", "A7o"
        }

    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state["street"]

        if street == "preflop":
            position = self._get_position(round_state)
            hand_key = self._get_hand_key(hole_card)

            if position == "SB":
                if hand_key in self.open_range:
                    return self._raise(valid_actions)
                else:
                    return self._fold(valid_actions)
            else:  # BB
                if hand_key in self.call_range:
                    return self._call(valid_actions)
                else:
                    return self._fold(valid_actions)
        else:
            return self._call(valid_actions)  # postflop: default to call

    def _get_position(self, round_state):
        return "SB" if self.first_player == self.uuid else "BB"

    def _get_hand_key(self, hole_card):
        rank1, suit1 = hole_card[0][1], hole_card[0][0]
        rank2, suit2 = hole_card[1][1], hole_card[1][0]

        if rank1 == rank2:
            return rank1 + rank2

        high, low = sorted([rank1, rank2], reverse=True)
        suited = suit1 == suit2
        return high + low + ("s" if suited else "o")

    def _call(self, valid_actions):
        action = [a for a in valid_actions if a["action"] == "call"][0]
        return action["action"], action["amount"]

    def _raise(self, valid_actions): # raise minimum amount
        action = [a for a in valid_actions if a["action"] == "raise"][0]
        return action["action"], action["amount"]["min"]

    def _fold(self, valid_actions):
        action = [a for a in valid_actions if a["action"] == "fold"][0]
        return action["action"], action["amount"]

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
