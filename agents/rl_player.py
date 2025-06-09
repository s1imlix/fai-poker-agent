import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from game.players import BasePokerPlayer

# D3QN Components
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
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
        features = self.feature_layer(x)
        state_value = self.value_stream(features)
        action_advantages = self.advantage_stream(features)
        q_values = state_value + (action_advantages - action_advantages.mean(dim=1, keepdim=True))
        return q_values

class D3QNPokerPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        # D3QN Parameters
        self.state_dim = 12
        self.action_dim = 3
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 10000
        self.batch_size = 64
        self.target_update_freq = 100
        
        # Networks and training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Experience replay
        self.replay_buffer = deque(maxlen=self.memory_size)
        self.update_counter = 0
        
        # Game state tracking
        self.last_state = None
        self.last_action = None
        self.round_count = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        # Extract state features
        state = self._extract_state(hole_card, round_state)
        
        # Select action using D3QN
        action_idx = self._select_action(state, valid_actions)
        
        # Store for learning
        self.last_state = state
        self.last_action = action_idx
        
        # Return action in required format
        selected_action = valid_actions[action_idx]
        return selected_action["action"], selected_action["amount"]

    def _extract_state(self, hole_card, round_state):
        """Extract numerical features from game state"""[1][2]
        features = []
        
        # Hand strength (simplified)
        hand_strength = self._calculate_hand_strength(hole_card, round_state.get('community_card', []))
        features.append(hand_strength)
        
        # Pot information
        pot_size = round_state['pot']['main']['amount']
        features.append(min(pot_size / 1000.0, 1.0))  # Normalized pot size
        
        # Street information
        community_cards = round_state.get('community_card', [])
        features.extend([
            len(community_cards) / 5.0,  # Street progress
            1.0 if len(community_cards) == 0 else 0.0,  # Preflop
            1.0 if len(community_cards) == 3 else 0.0,  # Flop
            1.0 if len(community_cards) == 4 else 0.0,  # Turn
            1.0 if len(community_cards) == 5 else 0.0,  # River
        ])
        
        # Position and round info
        features.extend([
            self.round_count / 100.0,  # Normalized round count
            len(round_state.get('seats', [])) / 10.0,  # Number of players
        ])
        
        # Betting information
        my_stack = 1000  # Default, get from game state if available
        for seat in round_state.get('seats', []):
            if seat.get('uuid') == getattr(self, 'uuid', None):
                my_stack = seat.get('stack', 1000)
                break
        
        features.extend([
            my_stack / 1000.0,  # Normalized stack
            pot_size / max(my_stack, 1) if my_stack > 0 else 0,  # Pot to stack ratio
        ])
        
        # Pad to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim], dtype=np.float32)

    def _calculate_hand_strength(self, hole_card, community_card):
        """Calculate basic hand strength"""[3]
        if not hole_card:
            return 0.5
        
        # Simple hand evaluation (can be improved with proper poker evaluation)
        card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        hole_values = [card_values.get(card[1], 7) for card in hole_card]
        avg_value = sum(hole_values) / len(hole_values)
        
        # Normalize to [0, 1]
        return (avg_value - 2) / 12

    def _select_action(self, state, valid_actions):
        """Epsilon-greedy action selection with valid action masking"""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(valid_actions))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                
                # Ensure we don't select invalid actions
                valid_q_values = q_values[:len(valid_actions)]
                return np.argmax(valid_q_values)

    def _train(self):
        """Train the D3QN network"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Learn from round outcome"""
        if self.last_state is not None and self.last_action is not None:
            # Calculate reward
            won = any(w.get('uuid') == getattr(self, 'uuid', None) for w in winners)
            reward = 1.0 if won else -1.0
            
            # Store experience
            next_state = np.zeros(self.state_dim)
            self.replay_buffer.append(Experience(
                self.last_state, self.last_action, reward, next_state, True
            ))
            
            # Train the network
            self._train()
        
        # Reset for next round
        self.last_state = None
        self.last_action = None

def setup_ai():
    return D3QNPokerPlayer()
