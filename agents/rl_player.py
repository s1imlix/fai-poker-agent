import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from game.players import BasePokerPlayer
from agents.card_utils import evaluate_hand, gen_cards, estimate_hole_card_win_rate
from agents.helper import street_to_id

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

class D3QNPokerPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        # For heads-up: state_dim can be small, e.g. [hand strength, pot size, stack sizes, street]
        self.state_dim = 6
        self.action_dim = 3  # fold, call, raise
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.5
        self.lr = 0.0001
        self.batch_size = 200
        self.memory_size = 5000
        self.target_update = 50
        self.max_raise_ratio = 0.1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=self.memory_size)
        self.steps = 0
        self.last_state = None
        self.last_action = None

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self._extract_state(round_state, hole_card)
        print(state)
        if np.random.rand() < self.epsilon:
            # Exploration: Random action with dynamic raise sizing
            action_idx = np.random.randint(len(valid_actions))
            action_info = valid_actions[action_idx]
            
            if action_info["action"] == "raise":
                # Random raise between min and max
                min_raise = action_info["amount"]["min"]
                max_raise = max(action_info["amount"]["max"] * self.max_raise_ratio, min_raise)
                amount = np.random.uniform(min_raise, max_raise)
            else:
                amount = action_info["amount"]
        else:
            # Exploitation: Use neural network
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(s).cpu().numpy()[0]
                
                # Mask invalid actions
                valid_q = [-np.inf] * len(valid_actions)
                for i, action in enumerate(valid_actions):
                    valid_q[i] = q_values[i]
                
                action_idx = np.argmax(valid_q)
                action_info = valid_actions[action_idx]
                
                if action_info["action"] == "raise":
                    # Calculate raise amount based on confidence
                    win_rate = state[0]
                    min_raise = action_info["amount"]["min"]
                    max_raise = max(action_info["amount"]["max"] * self.max_raise_ratio, min_raise)
                    exp_winrate = win_rate ** 10
                    amount = min_raise + (exp_winrate * (max_raise - min_raise))
                else:
                    amount = action_info["amount"]

        self.last_state = state
        self.last_action = action_idx
        print(f"Action: {action_info['action']}, Amount: {amount}")
        return action_info["action"], amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Heads-up: only two players, reward is win/loss
        if self.last_state is None or self.last_action is None:
            return
        my_uuid = getattr(self, 'uuid', None)
        reward = 1.0 if any(w.get('uuid') == my_uuid for w in winners) else -1.0
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        self.memory.append(Experience(self.last_state, self.last_action, reward, next_state, True))
        self._train()
        self.last_state = None
        self.last_action = None

    def _extract_state(self, round_state, hole_card):
        # Simple heads-up state: [hand strength, pot size, my stack, oppo stack, street, button?]

        hole_cards = gen_cards(hole_card)
        community_cards = gen_cards(round_state['community_card'])
        hand_strength = estimate_hole_card_win_rate(
                            nb_simulation=1000,
                            nb_player=2,
                            hole_card=hole_cards,
                            community_card=community_cards
                        )
        pot = round_state['pot']['main']['amount'] / 1000.0
        street = street_to_id[round_state['street']]
        button = 1.0 if self.first_player is None or self.first_player == self.uuid else 0.0
        return np.array([hand_strength, pot, self.my_stack, self.opp_stack, street / 5.0, button], dtype=np.float32)

    def _train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (~dones.unsqueeze(1))
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def receive_game_start_message(self, game_info):
        pass
    def receive_round_start_message(self, round_count, hole_card, seats):
        for player in seats:
            if player["uuid"] == self.uuid:
                self.my_stack = player["stack"]
            else:
                self.opp_stack = player["stack"]

        self.first_player = None

    def receive_street_start_message(self, street, round_state):
        pass
    def receive_game_update_message(self, action, round_state):
        if self.first_player is None:
            self.first_player = action["player_uuid"]

def setup_ai():
    return D3QNPokerPlayer()
