"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    count_x = sum(row.count(X) for row in board)
    count_o = sum(row.count(O) for row in board)

    if (count_x == count_o):
        return X
    else:
        return O



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    moves = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.add((i,j))

    return moves



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    current_player = player(board)

    new_board = [row[:] for row in board]
    new_board[action[0]][action[1]] = current_player

    return new_board



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
            return board[i][0]

    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] is not None:
            return board[0][j]

    if board[0][0] == board[1][1] == board[2][2] and board[1][1] is not None:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] and board[1][1] is not None:
        return board[1][1]

    return None



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) == O or winner(board) == X:
        return True

    if all(cell is not None for row in board for cell in row):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None

    def value(state):
        if terminal(state):
            return utility(state)

        current_player = player(state)
        is_maximizing = (current_player == X)

        if is_maximizing:
            best_value = -math.inf
            for action in actions(state):
                best_value = max(best_value, value(result(state, action)))
            return best_value
        else:
            best_value = math.inf
            for action in actions(state):
                best_value = min(best_value, value(result(state, action)))
            return best_value

    current_player = player(board)
    is_maximizing = (current_player == X)
    best_value = -math.inf if is_maximizing else math.inf
    best_action = None

    for action in actions(board):
        move_value = value(result(board, action))

        if is_maximizing:
            if move_value > best_value:
                best_value = move_value
                best_action = action
        else:
            if move_value < best_value:
                best_value = move_value
                best_action = action

    return best_action
