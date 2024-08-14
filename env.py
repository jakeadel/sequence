import random
from copy import deepcopy
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SequenceEnv(gym.Env):
    def __init__(self):
        super(SequenceEnv, self).__init__()

        self.num_players = 2
        
        self.board = np.array([
            ["WILD", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "WILD"],
            ["C6", "C5", "C4", "C3", "C2", "HA", "HK", "HQ", "H10", "S10"],
            ["C7", "SA", "D2", "D3", "D4", "D5", "D6", "D7", "H9", "SQ"],
            ["C8", "SK", "C6", "C5", "C4", "C3", "C2", "D8", "H8", "SK"],
            ["C9", "SQ", "C7", "H6", "H5", "H4", "HA", "D9", "H7", "SA"],
            ["C10", "S10", "C8", "H7", "H2", "H3", "HK", "D10", "H6", "D2"],
            ["CQ", "S9", "C9", "H8", "H9", "H10", "HQ", "DQ", "H5", "D3"],
            ["CK", "S8", "C10", "CQ", "CK", "CA", "DA", "DK", "H4", "D4"],
            ["CA", "S7", "S6", "S5", "S4", "S3", "S2", "H2", "H3", "D5"],
            ["WILD", "DA", "DK", "DQ", "D10", "D9", "D8", "D7", "D6", "WILD"]
        ])

        # 7 cards per hand
        card_action_space = spaces.Discrete(7)

        # 100 grid positions
        position_action_space = spaces.Discrete(100)

        self.action_space = spaces.Tuple((card_action_space, position_action_space))

        board_observation_space = spaces.MultiDiscrete([4] * 100)  # Using 4 values (0, 1, 2, 3) to represent empty, Player-owned, opponent-owned, and wild spaces

        hand_observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=int)

        self.observation_space = spaces.Tuple((board_observation_space, hand_observation_space))

    def reset(self):
        self.deck = Deck()
        self.board_state = np.zeros((10, 10), dtype=int)
        wild_positions = [(0, 0), (0, 9), (9, 0), (9, 9)]
        for row, col in wild_positions:
            self.board_state[row, col] = 3

        hands = []
        for _ in range(self.num_players):
            hand = []
            for _ in range(7):
                hand.append(self.deck.deal())
            hands.append(hand)

        return  deepcopy((self.board_state, hands[0])), deepcopy((self.board_state, hands[1]))

    def step(self, action, player):
        # Take the action and update the game state

        player_id = player.id
        self.apply_action(action, player_id)
        print(player_id)
        done = False

        fresh_card = self.deck.deal()
        if fresh_card == None:
            return None
        player.update_hand(action[1], fresh_card)
        fresh_observation = deepcopy((self.board_state, player.hand))

        return fresh_observation, done


    def check_sequences(self, player):

        def check_line(line, fresh_counts):

            # Check if there is a blocking move
            for i in range(6):
                window = line[i:i+5]
                if len(window) < 5:
                    return
                # Check for blockers
                for num in window:
                    if num in player.enemy_ids:
                        break
                # how many good in a row in this window
                count = 0
                for x in window:
                    if x == player.id or x == 3:
                        count += 1
                if count > 0:
                    if count in fresh_counts:
                        fresh_counts[count] += 1
                    else:
                        fresh_counts[count] = 1
            return

        def check_sequence(line):
            same_and_wildcards = [x == player.id or x == 3 for x in line]
            continuous_count = 0
            for i in range(len(same_and_wildcards)):
                if same_and_wildcards[i]:
                    continuous_count += 1
                else:
                    if continuous_count == 10:
                        return 2
                    elif continuous_count >= 5:
                        return 1
                    continuous_count = 0
            if continuous_count == 10:
                print("SEQUENCE-LINE DOUBLE", line)
                return 2
            elif continuous_count >= 5:
                print("SEQUENCE-LINE", line)
                return 1

            return 0

        lines = self.get_all_lines()
        fresh_counts = {}
        num_sequences_found = 0
        for line in lines:
            num_sequences_found += check_sequence(line)
            check_line(line, fresh_counts)
        
        player.sequences = num_sequences_found
        player.counts = deepcopy(fresh_counts)

    
    def get_all_lines(self):
        lines = []
        # Check rows
        for row in range(10):
            lines.append(self.board_state[row, :])

        # Check columns
        for col in range(10):
            lines.append(self.board_state[:, col])

        # Check diagonals
        for offset in range(-5, 6):
            diag1 = np.diagonal(self.board_state, offset=offset)
            diag2 = np.diagonal(np.fliplr(self.board_state), offset=offset)
            if len(diag1) >= 5:
                lines.append(diag1)
            if len(diag2) >= 5:
                lines.append(diag2)

        return lines


    def apply_action(self, action, player_id):
        move_type = action[0]
        location = action[2]
        if move_type == "place":
            self.board_state[location] = player_id
        elif move_type == "remove":
            self.board_state[location] = 0

    def _get_legal_actions(self, player):
        # Get the legal actions for the current state
        hand = player.hand
        has_one_eye = player.has_one_eye()
        has_two_eye = player.has_two_eye()
        actions = []
        for i, row in enumerate(self.board_state):
            for j, _ in enumerate(row):
                card = self.board[i, j]
                # If the card is in hand or wild card. And the space is open
                space_state = self.board_state[i, j]
                if space_state == 0:
                    if card in hand:
                        actions.append(("place", card, (i, j)))
                    elif has_two_eye:
                        actions.append(("place", "J2", (i, j)))
                elif has_one_eye and space_state in [1, 2] and space_state != player.id:
                    actions.append(("remove", "J1", (i, j)))

        return actions

    def render(self, mode='human'):
        print(self.board_state)
        print(len(self.deck.available), "cards left!!")
        return


class Deck:
    def __init__(self):
        self.cards = [
            "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "J1", "SQ", "SK", "SA",
            "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "J1", "HQ", "HK", "HA",
            "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "J1", "DQ", "DK", "DA",
            "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "J1", "CQ", "CK", "CA",
            "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "J2", "SQ", "SK", "SA",
            "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "J2", "HQ", "HK", "HA",
            "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "J2", "DQ", "DK", "DA",
            "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "J2", "CQ", "CK", "CA",
        ]

        self.available = self.cards
        
        self.dealt = []

    def deal(self):
        if len(self.available) == 0:
            print("No cards left!!!")
            return None
        random_index = random.randint(0, len(self.available) - 1)
        dealt_card = self.available.pop(random_index)
        self.dealt.append(dealt_card)
        return dealt_card
