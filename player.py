import random
from copy import deepcopy
import numpy as np


class Player:
    def __init__(self, id_in, enemy_ids_in):
        self.hand = []
        self.id = id_in
        self.enemy_ids = enemy_ids_in
        self.move_history = []
        self.sequences = 0
        self.counts = {} # Dictionary of num in a row in a stretch of five squares (does not have to be continuous)
        # in the form counts[num of already placed pieces/wildcards] = occurences of this num within a window

    def choose_action(self, env, observation):
        legal_actions = env._get_legal_actions(self)
        if legal_actions:
            choice = random.choice(legal_actions)
            return choice
        else:
            return None

    def learn(self, experience, env):
        pass

    def state_to_str(self, state):
        board_str = ''.join([''.join([str(cell) for cell in row]) for row in state[0]])
        hand_str = ''.join(state[1])
        return board_str + hand_str

    def action_to_index(self, action):
        move_type, card_str, board_pos = action
        move_type_index = 0 if move_type == "place" else 1
        card_index = np.argmax(self.card_to_one_hot(card_str))
        board_index = board_pos[0] * 10 + board_pos[1]
        return (move_type_index, card_index, board_index)

    def card_to_one_hot(self, card_str):
        suits = 'CDHS'
        ranks = 'A23456789QK'
        one_hot = np.zeros(52)
        if card_str == "J1":
            one_hot[50] = 1
        elif card_str == "J2":
            one_hot[51] = 2
        else:
            suit_idx = suits.index(card_str[0])
            rank_idx = ranks.index(card_str[1]) if len(card_str) == 2 else 9
            one_hot[suit_idx * (len(ranks) + 2) + rank_idx] = 1

        return one_hot

    def has_one_eye(self):
        for card in self.hand:
            if card == "J1":
                return True
        return False

    def has_two_eye(self):
        for card in self.hand:
            if card == "J2":
                return True
        return False
    
    def update_hand(self, out_card, in_card):
        fresh_hand = deepcopy(self.hand)
        fresh_hand.remove(out_card)
        fresh_hand.append(in_card)
        self.hand = fresh_hand

    def set_hand(self, hand):
        self.hand = hand
