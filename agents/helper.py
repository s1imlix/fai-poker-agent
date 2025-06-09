import numpy as np

def sigmoid(x, sharpness=5, midpoint=0.5):
    return 1 / (1 + np.exp(-sharpness * (x - midpoint)))


open_range = {
    # Pocket pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # Suited hands
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s",
    "JTs", "J9s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s",
    "T9s", "T8s", "T7s", "T6s", "T5s", "T4s", "T3s", "T2s",
    "98s", "97s", "96s", "95s", "94s", "93s", "92s",
    "87s", "86s", "85s", "84s", "83s", "82s",
    "76s", "75s", "74s", "73s", "72s",
    "65s", "64s", "63s", "62s",
    "54s", "53s", "52s",
    "43s", "42s",
    "32s",
    # Offsuit hands
    "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
    "KQo", "KJo", "KTo", "K9o", "K8o", "K7o", "K6o", "K5o", "K4o", "K3o", "K2o",
    "QJo", "QTo", "Q9o", "Q8o", "Q7o", "Q6o", "Q5o", "Q4o", "Q3o", "Q2o",
    "JTo", "J9o", "J8o", "J7o", "J6o", "J5o", "J4o", "J3o", "J2o",
    "T9o", "T8o", "T7o", "T6o", "T5o", "T4o", "T3o", "T2o",
    "98o", "97o", "96o", "95o", "94o", "93o", 
    "87o", "86o", "85o", "84o", 
    "76o", "75o", "74o", 
    "65o", "64o", "63o", 
    "54o", "53o", 
}

call_range = {
    # Pocket pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # Suited hands
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s",
    "JTs", "J9s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s",
    "T9s", "T8s", "T7s", "T6s", "T5s", "T4s", "T3s", "T2s",
    "98s", "97s", "96s", "95s", "94s", "93s", 
    "87s", "86s", "85s", "84s", 
    "76s", "75s", "74s", 
    "65s", "64s", "63s", 
    "54s", "53s", 
    "43s", 
    # Offsuit hands
    "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
    "KQo", "KJo", "KTo", "K9o", "K8o", "K7o", "K6o", "K5o", "K4o", "K3o", "K2o",
    "QJo", "QTo", "Q9o", "Q8o", "Q7o", "Q6o", "Q5o", "Q4o", "Q3o", 
    "JTo", "J9o", "J8o", "J7o", "J6o", "J5o", 
    "T9o", "T8o", "T7o", "T6o", 
    "98o", "97o", "96o", 
    "87o", "86o", 
    "76o", 
}


def _get_position(self, round_state, first_player=None):
        return "SB" if first_player == None else "BB"

def _get_hand_key(hole_card):
    rank1, suit1 = hole_card[0][1], hole_card[0][0]
    rank2, suit2 = hole_card[1][1], hole_card[1][0]

    if rank1 == rank2:
        return rank1 + rank2

    high, low = sorted([rank1, rank2], reverse=True)
    suited = suit1 == suit2
    return high + low + ("s" if suited else "o")

def _call(valid_actions):
    return valid_actions[1]["action"], valid_actions[1]["amount"]

def _raise(valid_actions): 
    return valid_actions[2]["action"], 15 
def _fold(valid_actions):
    return valid_actions[0]["action"], valid_actions[0]["amount"]

def preflop_table_lookup(valid_actions, hole_card, round_state, first_player):
    street = round_state["street"]

    if street == "preflop":
        position = "SB" if first_player == None else "BB"
        hand_key = _get_hand_key(hole_card)
        print(position, hand_key)
        if position == "SB":
            if hand_key in open_range:
                action, amount = _raise(valid_actions)
            else:
                action, amount = _fold(valid_actions)
        else:  # BB
            if hand_key in call_range:
                action, amount = _call(valid_actions)
            else:
                action, amount = _fold(valid_actions)
    else:
        action, amount = _call(valid_actions)  # postflop: default to call
    print(f"Table: pos={position}, hand={hand_key} -> action={action}, amount={amount}")
    return action, amount