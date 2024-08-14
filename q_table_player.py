from player import Player
from copy import deepcopy
import random
import numpy as np


class Q_player(Player):
    def __init__(self, id_in, enemy_ids_in, use_model=False, alpha=0.1, gamma=0.99, epsilon=0.8):
        super().__init__(id_in, enemy_ids_in)
        self.hand = []
        self.use_model = use_model
        self.move_history = []

        if self.use_model:
            self.q_table = {}
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon

    def choose_action(self, env, observation):
        legal_actions = env._get_legal_actions(self)
        if legal_actions:
            if self.use_model and random.uniform(0, 1) > self.epsilon:
                q_values = [self.get_q_value(observation, action) for action in legal_actions]

                max_q_index = np.argmax(q_values)
                #self.print_action_info(env, observation, legal_actions, q_values)
                return legal_actions[max_q_index]
            else:
                choice = random.choice(legal_actions)
                return choice
        else:
            return None

    def learn(self, experience, env):
        if not self.use_model:
            return

        observation, action, reward, next_observation = experience
        max_q_next = np.max([self.get_q_value(next_observation, a) for a in env._get_legal_actions(self)])

        q_value = self.get_q_value(observation, action)
        target_q = reward + self.gamma * max_q_next
        updated_q = q_value + self.alpha * (target_q - q_value)

        self.update_q_value(observation, action, updated_q)
        self.move_history.append((deepcopy(observation), action))

        if reward == 10 or reward == 5:  # Recognizes a completed sequence
            for i in range(1, min(11, len(self.move_history))):
                prev_observation, prev_action = self.move_history[-i-1]
                print(self.state_to_str(prev_observation))
                prev_q_value = self.get_q_value(prev_observation, prev_action)
                prev_target_q = (reward / (i+1)) + self.gamma * max_q_next
                prev_updated_q = prev_q_value + self.alpha * (prev_target_q - prev_q_value)
                self.update_q_value(prev_observation, prev_action, prev_updated_q)

    def get_q_value(self, state, action):
        state_str = self.state_to_str(state)
        state_action = (state_str, self.action_to_index(action))
        return self.q_table.get(state_action, 0)

    def update_q_value(self, state, action, value):
        state_str = self.state_to_str(state)
        state_action = (state_str, self.action_to_index(action))
        self.q_table[state_action] = value

    def print_action_info(self, env, observation, legal_actions, q_values):
        print("Board:")
        print(observation[0])
        print("Board State:", observation[1])
        print("Player's hand:", self.hand)

        print("Legal actions and their Q-values:")
        for action, q_value in zip(legal_actions, q_values):
            print(f"Action: {action}, Q-value: {q_value}")
