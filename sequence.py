import time
from env import SequenceEnv
from player import Player
from q_table_player import Q_player


if __name__ == "__main__":

    num_episodes = 10000

    env = SequenceEnv()
    player1 = Q_player(1, [2], True)
    player2 = Player(2, [1])
    reward1 = 0
    reward2 = 0

    # Training loop
    for episode in range(num_episodes):

        observation1, observation2 = env.reset()
        player1.set_hand(observation1[1])
        player1.move_history = []
        player2.set_hand(observation2[1])
        player2.move_history = []
        player1.sequences = 0
        player2.sequences = 0
        player1.counts = {}
        player2.counts = {}
        done = False

        while not done:
            # player 1 takes an action
            action1 = player1.choose_action(env, observation1)
            try:
                next_observation1, done = env.step(action1, player1)
                env.render()
            except TypeError:
                # No cards left!!
                done = True
                break

            current_sequences = player1.sequences
            env.check_sequences(player1)
            fresh_sequences = player1.sequences

            if fresh_sequences >= 2:
                reward = 10
                done = True
            elif fresh_sequences > current_sequences:
                reward = 5
            else:
                reward = 0

            reward1 += reward
            player1.learn((observation1, action1, reward, next_observation1), env)
            observation1 = next_observation1

            # Check if the game is already over before the second player's turn
            if done:
                env.render()
                print("1 Wins!!")
                break

            # player 2 takes an action
            action2 = player2.choose_action(env, observation2)
            try:
                next_observation2, done = env.step(action2, player2)
                env.render()
            except TypeError:
                # No cards left!!
                done = True
                break

            current_sequences = player2.sequences
            env.check_sequences(player2)
            fresh_sequences = player2.sequences

            if fresh_sequences >= 2:
                reward = 10
                done = True
            elif fresh_sequences > current_sequences:
                reward = 5
            else:
                reward = 0

            reward2 += reward
            player2.learn((observation2, action2, reward, next_observation2), env)
            observation2 = next_observation2

            # Check if the game is already over before the second player's turn
            if done:
                print("2 Wins!!")
                break

        print("Current rewards, 1:", reward1, "2:", reward2)
