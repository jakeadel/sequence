import numpy as np

def check_sequences(board, player_id):
    def count_sequences(board, player_id):
        count = 0

        def check_line(line):
            nonlocal count
            same_and_wildcards = [x == player_id or x == 3 for x in line]
            continuous_count = 0
            for i in range(len(same_and_wildcards)):
                if same_and_wildcards[i]:
                    continuous_count += 1
                else:
                    if continuous_count == 10:
                        count += 2
                    elif continuous_count >= 5:
                        count += 1
                    continuous_count = 0
            if continuous_count == 10:
                count += 2
            elif continuous_count >= 5:
                count += 1

        # Check rows
        for row in range(10):
            check_line(board[row, :])

        # Check columns
        for col in range(10):
            check_line(board[:, col])

        # Check diagonals
        for offset in range(-5, 6):
            diag1 = np.diagonal(board, offset=offset)
            diag2 = np.diagonal(np.fliplr(board), offset=offset)
            if len(diag1) >= 5:
                check_line(diag1)
            if len(diag2) >= 5:
                check_line(diag2)

        return count

    sequences = {}
    sequences[1] = count_sequences(board, 1)
    sequences[2] = count_sequences(board, 2)
    return sequences

test_board1 = np.array([
    [1, 1, 1, 1, 1, 2, 0, 3, 0, 0],
    [0, 1, 1, 0, 0, 3, 0, 3, 0, 0],
    [0, 1, 0, 1, 0, 3, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 2, 0, 2, 0, 0],
    [0, 1, 0, 0, 0, 3, 0, 3, 0, 0],
    [1, 1, 1, 1, 1, 2, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0, 0]
])

print(check_sequences(test_board1, 1))
