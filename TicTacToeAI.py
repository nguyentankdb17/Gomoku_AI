import random

RANGE = 5


class Board:
    def __init__(self, vertical, horizontal, verbose=False):
        self.state = [[' ' for _ in range(horizontal)] for _ in range(vertical)]
        self.eval = [[0 for _ in range(horizontal)] for _ in range(vertical)]
        self.xwins = 0
        self.owins = 0
        self.draws = 0
        self._num_games = 0
        self.verbose = verbose
        self.rows = vertical
        self.cols = horizontal
        self.bother = set()

    def __getitem__(self, index):
        return self.state[index]

    def __setitem__(self, key, value):
        self.state[key] = value

    def __str__(self):
        result = ""
        result += "   " + "   ".join(str(i) for i in range(self.cols)) + "\n"
        for i, row in enumerate(self.state):
            result += str(i) + "  " + " | ".join(cell for cell in row) + "\n"
            if i < self.rows - 1:
                result += "  " + "-" * (4 * self.cols - 1) + "\n"
        return result

    def num_games(self):
        self._num_games = self.xwins + self.owins + self.draws
        return self._num_games

    def getstate(self):
        return tuple(tuple(a) for a in self.state)

    def reset(self):
        self.state = [[' ' for _ in range(self.cols)] for _ in range(self.rows)]

    def check_win(self, player):
        def check_line(line):
            count = 0
            for cell in line:
                if cell == player:
                    count += 1
                    if count == 5:
                        return True
                else:
                    count = 0
            return False

        # Check rows
        for i in range(self.rows):
            if check_line(self.state[i]):
                return True

        # Check columns
        for j in range(self.cols):
            column = [self.state[i][j] for i in range(self.rows)]
            if check_line(column):
                return True

        # Check diagonals
        for i in range(self.rows - 4):
            for j in range(self.cols - 4):
                diagonal = [self.state[i + k][j + k] for k in range(5)]
                if check_line(diagonal):
                    return True
                diagonal = [self.state[i + 4 - k][j + k] for k in range(5)]
                if check_line(diagonal):
                    return True

        return False

    def check_draw(self):
        return all(self.state[i][j] != ' ' for i in range(self.cols) for j in range(self.rows))

    def check_gameover(self):
        if self.check_win('o'):
            if self.verbose:
                print("o wins!")
            self.owins += 1
            return True
        if self.check_win('x'):
            if self.verbose:
                print("x wins!")
            self.xwins += 1
            return True
        if self.check_draw():
            if self.verbose:
                print("Draw!")
            self.draws += 1
            return True
        return False


heuristic = [[0, -0.25, -0.5, -0.75, -1],
             [0.25, 0, -0.25, -0.5, 0],
             [0.5, 0.25, 0, 0, 0],
             [0.75, 0.5, 0, 0, 0],
             [1, 0, 0, 0, 0]]


def evaluate(board, player, opponent):
    score = 0

    # Calculate the score in rows and cols
    for i in range(board.rows):
        player_score_row = sum(board[i][j] == player for j in range(board.cols))
        opponent_score_row = sum(board[i][j] == opponent for j in range(board.cols))
        player_score_col = sum(board[j][i] == player for j in range(board.cols))
        opponent_score_col = sum(board[j][i] == opponent for j in range(board.cols))
        score += heuristic[player_score_row][opponent_score_row]
        score += heuristic[player_score_col][opponent_score_col]

    # Calculate the score in main diagonal
    player_score_diag_main = sum(board[i][i] == player for i in range(RANGE))
    opponent_score_diag_main = sum(board[i][i] == player for i in range(RANGE))
    score += heuristic[player_score_diag_main][opponent_score_diag_main]

    # Calculate the score in secondary diagonal
    player_score_diag_secondary = sum(board[i][RANGE - 1 - i] == player for i in range(RANGE))
    opponent_score_diag_secondary = sum(board[i][RANGE - 1 - i] == opponent for i in range(RANGE))
    score += heuristic[player_score_diag_secondary][opponent_score_diag_secondary]

    return score


def minimax(board, player, players, max, depth):
    if players[0] == player:
        other_player = players[1]
    else:
        other_player = players[0]

    if board.check_win(player):
        return 1
    if board.check_draw():
        return 0
    if board.check_win(other_player):
        return -1

    if depth == 0:
        score = evaluate(board, player, other_player)
        return score

    best_score = float('-inf')

    for (i, j) in board.bother:
        if best_score >= max:
            return best_score
        board[i][j] = player
        subscore = - minimax(board, other_player, players, - best_score, depth - 1)
        board[i][j] = ' '
        if subscore > best_score:
            best_score = subscore

    return best_score


def find_move(board, player, players, depth):
    if players[0] == player:
        other_player = players[1]
    else:
        other_player = players[0]

    best_score, move = float('-inf'), (0, 0)
    DEP = 1
    values = []
    while True:
        for (i, j) in board.bother:
            if board[i][j] == ' ':
                board[i][j] = player
                subscore = - minimax(board, other_player, players, - best_score, depth)
                board.heuristic[i][j] = subscore
                board[i][j] = ' '
                if subscore > best_score:
                    move = (i, j)
                    best_score = subscore
            else:
                values.append(-10)
        DEP += 1
        if DEP > depth:
            break
    return move, values


def make_move(board, player, players, depth):
    if board.check_gameover():
        return -1
    moves = []
    for (row, col) in board.bother:
        moves.append((row, col))
    move = random.choice(moves)
    # move, values = find_move(board, player, players, depth)
    board[move[0]][move[1]] = player
    add_bother(board, move[0], move[1])
    board.eval[move[0]][move[1]] = -10


def is_size_valid(vertical, horizontal):
    return vertical >= 5 and horizontal >= 5


def is_move_valid(board, row, col):
    return 0 <= row < board.rows and 0 <= col < board.cols and board[row][col] == ' '


def add_bother(board, row, col):
    if (row, col) in board.bother:
        board.bother.remove((row, col))
    consider = [(row - 2, col - 2), (row - 2, col), (row - 2, col + 2), (row - 1, col - 1),
                (row - 1, col), (row - 1, col + 1), (row, col - 2), (row, col - 1),
                (row, col + 1), (row, col + 2), (row + 1, col - 1), (row + 1, col),
                (row + 1, col + 1), (row + 2, col - 2), (row + 2, col), (row + 2, col + 2)]

    for (i, j) in consider:
        if is_move_valid(board, i, j):
            board.bother.add((i, j))


def play_game():
    players = ['x', 'o']
    board = Board(0, 0, verbose=True)
    while True:
        size = input("Enter the board size : ").split()
        if len(size) != 2:
            print("Invalid input! Please enter 2 numbers separated by space.")
            continue
        try:
            vertical, horizontal = map(int, size)
        except ValueError:
            print("Invalid input! Please enter valid integers.")
            continue
        if is_size_valid(vertical, horizontal):
            board = Board(vertical, horizontal, verbose=True)
            print(board)
            break
        else:
            print("Invalid size! Please enter the size greater than 5 !")
            continue

    while True:
        moves = input("Your move: ").split()
        if len(moves) != 2:
            print("Invalid input! Please enter row and column separated by space.")
            continue
        try:
            row, col = map(int, moves)
        except ValueError:
            print("Invalid input! Please enter valid integers.")
            continue

        if is_move_valid(board, row, col):
            board[row][col] = 'o'
            board.eval[row][col] = -10
            add_bother(board, row, col)
            print()
            print(board)
        else:
            print("Invalid move! Please select an empty cell.")
            continue

        print("Bot is thinking....")
        print()

        if make_move(board, 'x', players, 5) == -1:
            break

        print(board)

        if board.check_gameover():
            break


if __name__ == "__main__":
    while True:
        play_game()
        print("To quit, press q. To play again, press any key.")
        if input() == 'q':
            break
