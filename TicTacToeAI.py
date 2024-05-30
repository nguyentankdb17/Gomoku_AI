import time
import numpy as np
import random


class Board:
    def __init__(self, size, room_id, match_id, team1_id="xx1+x", team2_id="xx2+o"):
        self.size = size
        self.status = None
        self.team1_time = 0
        self.team2_time = 0
        self.score1 = 0
        self.score2 = 0
        self.board = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.game_info = {
            "room_id": room_id,
            "match_id": match_id,
            "status": self.status,
            "size": size,
            "board": self.board,
            "time1": self.team1_time,
            "time2": self.team2_time,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "turn": team1_id,
            "score1": self.score1,
            "score2": self.score2
        }
        self.timestamps = [time.time()] * 2
        self.start_game = False

    def is_win(self, board):
        # new_board = self.convert_board(board)
        new_board = board
        # print("New board: ", new_board)
        black = self.score_of_col(new_board, 'x')
        white = self.score_of_col(new_board, 'o')

        self.sum_sumcol_values(black)
        self.sum_sumcol_values(white)

        if 5 in black and black[5] == 1:
            return 'X won'
        elif 5 in white and white[5] == 1:
            return 'O won'

        if sum(black.values()) == black[-1] and sum(white.values()) == white[-1] or self.possible_moves(board) == []:
            return 'Draw'

        return 'Continue playing'

    def check_status(self, board):
        win_check = self.is_win(board)
        if win_check == 'X won' or win_check == 'O won':
            self.status = win_check
            self.game_info["status"] = self.status
            print("Result: " + win_check)
            return
        elif win_check == 'Draw':
            flag = True
            # Check if there is no free space
            for i in range(self.size):
                for j in range(self.size):
                    if board[i][j] == " ":
                        flag = False

            if flag:
                if self.game_info["time1"] < self.game_info["time2"]:
                    self.status = self.game_info["team1_id"][-1].upper() + " won"
                elif self.game_info["time1"] > self.game_info["time2"]:
                    self.status = self.game_info["team2_id"][-1].upper() + " won"
                self.game_info["status"] = self.status
                print("Draw and compare time: " + self.status)
                return
        # else:
        #     print("Continue playing")

    def make_empty_board(self, sz):
        board = []
        for i in range(sz):
            board.append([" "] * sz)
        return board

    def is_empty(self, board):
        return board == [[' '] * len(board)] * len(board)

    def is_in(self, board, y, x):
        return 0 <= y < len(board) and 0 <= x < len(board)

    def score_ready(self, scorecol):
        '''
        Khởi tạo hệ thống điểm

        '''
        sumcol = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, -1: {}}
        for key in scorecol:
            for score in scorecol[key]:
                if key in sumcol[score]:
                    sumcol[score][key] += 1
                else:
                    sumcol[score][key] = 1

        return sumcol

    def sum_sumcol_values(self, sumcol):
        '''
        hợp nhất điểm của mỗi hướng
        '''

        for key in sumcol:
            if key == 5:
                sumcol[5] = int(1 in sumcol[5].values())
            else:
                sumcol[key] = sum(sumcol[key].values())

    def score_of_col(self, board, col):
        '''
        tính toán điểm số mỗi hướng của column dùng cho is_win;
        '''

        f = len(board)
        # scores của 4 hướng đi
        scores = {(0, 1): [], (-1, 1): [], (1, 0): [], (1, 1): []}
        for start in range(len(board)):
            scores[(1, 0)].extend(self.score_of_row(board, (0, start), 1, 0, (f - 1, start), col))
            scores[(0, 1)].extend(self.score_of_row(board, (start, 0), 0, 1, (start, f - 1), col))
            scores[(1, 1)].extend(self.score_of_row(board, (start, 0), 1, 1, (f - 1, f - 1 - start), col))
            scores[(-1, 1)].extend(self.score_of_row(board, (start, 0), -1, 1, (0, start), col))

            if start + 1 < len(board):
                scores[(1, 1)].extend(self.score_of_row(board, (0, start + 1), 1, 1, (f - 2 - start, f - 1), col))
                scores[(-1, 1)].extend(self.score_of_row(board, (f - 1, start + 1), -1, 1, (start + 1, f - 1), col))

        return self.score_ready(scores)

    def score_of_list(self, lis, col):
        blank = lis.count(' ')
        filled = lis.count(col)

        if blank + filled < 5:
            return -1
        elif blank == 5:
            return 0
        else:
            return filled

    def score_of_row(self, board, cordi, dy, dx, cordf, col):
        '''
        trả về một list với mỗi phần tử đại diện cho số điểm của 5 khối

        '''
        colscores = []
        y, x = cordi
        yf, xf = cordf
        row = self.row_to_list(board, y, x, dy, dx, yf, xf)
        for start in range(len(row) - 4):
            score = self.score_of_list(row[start:start + 5], col)
            colscores.append(score)

        return colscores

    def possible_moves(self, board):
        '''
        khởi tạo danh sách tọa độ có thể có tại danh giới các nơi đã đánh phạm vi 3 đơn vị
        '''
        # mảng taken lưu giá trị của người chơi và của máy trên bàn cờ
        taken = []
        # mảng directions lưu hướng đi (8 hướng)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        # cord: lưu các vị trí không đi
        cord = {}

        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] != ' ':
                    taken.append((i, j))
        ''' duyệt trong hướng đi và mảng giá trị trên bàn cờ của người chơi và máy, kiểm tra nước không thể đi(trùng với 
        nước đã có trên bàn cờ)
        '''
        for direction in directions:
            dy, dx = direction
            for coord in taken:
                y, x = coord
                for length in [1, 2, 3, 4]:
                    move = self.march(board, y, x, dy, dx, length)
                    if move not in taken and move not in cord:
                        cord[move] = False
        return cord

    def row_to_list(self, board, y, x, dy, dx, yf, xf):
        """
        trả về list của y,x từ yf,xf

        """
        row = []
        while y != yf + dy or x != xf + dx:
            row.append(board[y][x])
            y += dy
            x += dx
        return row

    def march(self, board, y, x, dy, dx, length):
        '''
        tìm vị trí xa nhất trong dy,dx trong khoảng length

        '''
        yf = y + length * dy
        xf = x + length * dx
        # chừng nào yf,xf không có trong board
        while not self.is_in(board, yf, xf):
            yf -= dy
            xf -= dx

        return yf, xf

    '''
    deprecated
    '''

    def convert_board(self, board):
        new_board = []
        for i in range(len(board)):
            row = []
            row.append(board[i][0])
            if i % self.size - 1 == ' ':
                new_board.append(row)
        return new_board

    def check_valid_move(self, new_move_pos):
        # Kiểm tra nước đi hợp lệ
        # Điều kiện đơn giản là ô trống mới có thể đánh vào
        if new_move_pos is None:
            return False
        i, j = int(new_move_pos[0]), int(new_move_pos[1])
        if i < self.size and j < self.size and self.board[i][j] == ' ':
            return True
        return False


# 1 represents player, 2 represents opponent, 0 represents empty cell
PLAYER_TERMINATE_STATE = {
    '11111': 5000000000,
    '011110': 200000000
}

PLAYER_FORK_STATE = {
    '011112': 500000,
    '211110': 500000,
    '01110': 30000,
    '011010': 15000,
    '010110': 15000,
}

PLAYER_NORMAL_STATE = {
    '001112': 2000,
    '211100': 2000,
    '211010': 2000,
    '210110': 2000,
    '010112': 2000,
    '011012': 2000,
    '01100': 500,
    '00110': 500
}

OPPONENT_TERMINATE_STATE = {
    '22222': -5000000000,
    '022220': -200000000
}

OPPONENT_FORK_STATE = {
    '022221': -500000,
    '122220': -500000,
    '02220': -30000,
    '022020': -15000,
    '020220': -15000
}

OPPONENT_NORMAL_STATE = {
    '002221': -2000,
    '122200': -2000,
    '122020': -2000,
    '120220': -2000,
    '020221': -2000,
    '022021': -2000,
    '02200': -500,
    '00220': -500
}

PLAYER_HEURISTIC = PLAYER_TERMINATE_STATE | PLAYER_FORK_STATE | PLAYER_NORMAL_STATE
OPPONENT_HEURISTIC = OPPONENT_TERMINATE_STATE | OPPONENT_FORK_STATE | OPPONENT_NORMAL_STATE
HEURISTIC = PLAYER_HEURISTIC | OPPONENT_HEURISTIC


def makeLine(matrix, type_list, player):
    for i in matrix:
        res = ""
        for j in i:
            if j == ' ':
                res += "0"
            elif j == player:
                res += "1"
            else:
                res += "2"
        type_list.append(res)


# Make coordinates
def convert_state(board, player):
    """
    convert board to col,row,and diag string arrays
    """
    newBoard = np.array(board)
    boardT = newBoard.copy().transpose()
    col_list, row_list, diag_list = [], [], []
    main_diag = [newBoard.diagonal(i) for i in range(newBoard.shape[1] - 5, -newBoard.shape[1] + 4, -1)]
    second_diag = [newBoard[::-1, :].diagonal(i) for i in range(newBoard.shape[1] - 5, -newBoard.shape[1] + 4, -1)]

    makeLine(main_diag, diag_list, player)
    makeLine(second_diag, diag_list, player)
    makeLine(newBoard, row_list, player)
    makeLine(boardT, col_list, player)

    return diag_list + col_list + row_list


def fork_value(board, is_player_turn, player):
    fork_val = 0
    size = len(board)
    possible_pos = [(i, j) for i in range(size) for j in range(size) if board[i][j] != ' ']
    if len(possible_pos) < 5:
        return 0

    for (row, col) in possible_pos:
        margin_left = max(0, col - 5)
        margin_right = min(size - 1, col + 5)
        margin_top = max(0, row - 5)
        margin_bottom = min(size - 1, row + 5)

        tmp_list = [
            [board[row][i] for i in range(margin_left, margin_right + 1)],
            [board[i][col] for i in range(margin_top, margin_bottom + 1)],
            [board[row + k][col + k] for k in range(-5, 6) if 0 <= row + k < size and 0 <= col + k < size],
            [board[row + k][col - k] for k in range(-5, 6) if 0 <= row + k < size and 0 <= col - k < size]
        ]

        lines = []
        makeLine(tmp_list, lines, player)
        count_player, count_opponent = 0, 0
        for line in lines:
            for i in range(len(line)):
                for j in [5, 6]:
                    if i + j <= len(line):
                        sub_line = line[i:i + j]
                        if sub_line in PLAYER_FORK_STATE:
                            count_player += 1
                        if sub_line in OPPONENT_FORK_STATE:
                            count_opponent += 1
        if is_player_turn:
            fork_val += (count_player - 1) * 4000000 * 1.25 if count_player > 0 else 0
            fork_val -= (count_opponent - 1) * 4000000 if count_opponent > 0 else 0
        else:
            fork_val += (count_player - 1) * 4000000 if count_player > 0 else 0
            fork_val -= (count_opponent - 1) * 4000000 * 1.25 if count_opponent > 0 else 0
    return fork_val


def value(board, is_player_turn, player):
    """
    calculate value for each move
    """
    val = 0
    lines = convert_state(board, player)
    for line in lines:
        for i in range(len(line)):
            for j in [5, 6]:
                if i + j <= len(line):
                    sub_line = line[i:i + j]
                    if sub_line in PLAYER_HEURISTIC:
                        val += HEURISTIC[sub_line] * 1.25 if is_player_turn else HEURISTIC[sub_line]
                    if sub_line in OPPONENT_HEURISTIC:
                        val += HEURISTIC[sub_line] if is_player_turn else HEURISTIC[sub_line] * 1.25
    val += fork_value(board, is_player_turn, player)
    return val


def getAround(board):
    """
    Get values around placed stones on the board.
    """

    consider = np.nonzero(board)
    potentialCoords = set()

    for x, y in zip(consider[0], consider[1]):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the current stone itself
                nx, ny = x + dx, y + dy
                if 0 <= ny < len(board) and 0 <= nx < len(board):
                    potentialCoords.add((nx, ny))

    potentialRows, potentialCols = zip(*potentialCoords) if potentialCoords else ([], [])
    return list(potentialRows), list(potentialCols)


# Use minimax and alpha-beta pruning algorithm
def minimax(board, is_maximizing, depth, alpha, beta, player):
    """
    Minimax with Alpha-Beta pruning
    """
    point = value(board, is_maximizing, player)
    if depth == 2 or point >= 200000000 or point <= -200000000:
        return value(board, is_maximizing, player)
    if is_maximizing:  # THE MAXIMIZER
        best = float('-inf')
        potentialRows, potentialCols = getAround(board)
        for i in range(len(potentialCols)):
            if board[potentialRows[i]][potentialCols[i]] == ' ':
                board[potentialRows[i]][potentialCols[i]] = player
                score = minimax(board, False, depth + 1, alpha, beta, player)
                best = max(best, score)
                alpha = max(alpha, best)  # best opponent move
                board[potentialRows[i]][potentialCols[i]] = ' '  # undoing
                if beta <= alpha:
                    break
        return best
    else:  # THE MINIMIZER
        best = float('inf')
        potentialRows, potentialCols = getAround(board)
        for i in range(len(potentialCols)):
            if board[potentialRows[i]][potentialCols[i]] == ' ':
                opponent = 'x' if player == 'o' else 'o'
                board[potentialRows[i]][potentialCols[i]] = opponent
                score = minimax(board, True, depth + 1, alpha, beta, player)
                best = min(best, score)
                beta = min(beta, best)  # best opponent move
                board[potentialRows[i]][potentialCols[i]] = ' '  # undoing
                if beta <= alpha:
                    break
        return best


# Get a random move
def getRandomMove(board):
    """
    Returns a random move around central position
    If no move near central found, return a random possible move
    """
    boardSize = len(board)
    ctr = 0
    idx = boardSize // 2
    while ctr < (idx / 2):
        ctr_list = [(idx + ctr, idx + ctr), (idx + ctr, idx - ctr), (idx + ctr, idx), (idx, idx + ctr),
                    (idx, idx - ctr), (idx - ctr, idx), (idx - ctr, idx - ctr), (idx - ctr, idx + ctr)]
        for (i, j) in ctr_list:
            if board[i][j] == ' ':
                return i, j
        ctr += 1

    # If no possible moves near central position choose a random possible move
    possible_moves = [(i, j) for i in range(boardSize) for j in range(boardSize) if board[i][j] == ' ']
    return random.choice(possible_moves)


# Get the move
def get_move(board, size, player):
    mostPoints = float('-inf')
    alpha, beta = float('-inf'), float('inf')
    bestMoveRow = bestMoveCol = -1
    potentialRows, potentialCols = getAround(board)
    for i in range(len(potentialCols)):
        if board[potentialRows[i]][potentialCols[i]] == ' ':
            board[potentialRows[i]][potentialCols[i]] = player
            movePoints = max(mostPoints, minimax(
                board, False, 1, alpha, beta, player))
            alpha = max(alpha, movePoints)
            board[potentialRows[i]][potentialCols[i]] = ' '
            if beta <= alpha:
                break
            if movePoints > mostPoints:
                bestMoveRow = potentialRows[i]
                bestMoveCol = potentialCols[i]
                mostPoints = movePoints
                if movePoints >= 200000000:
                    break
    if bestMoveRow == -1 or bestMoveCol == -1:
        bestMoveRow, bestMoveCol = getRandomMove(board)
    return bestMoveRow, bestMoveCol


def make_move(game_board, player):
    move = get_move(game_board.board, game_board.size, player)
    game_board.board[int(move[0])][int(move[1])] = player


def print_board(game_board):
    result = ""
    result += "   " + "   ".join(str(i) for i in range(game_board.size)) + "\n"
    for i, row in enumerate(game_board.board):
        result += str(i) + "  " + " | ".join(cell for cell in row) + "\n"
        if i < game_board.size - 1:
            result += "  " + "-" * (4 * game_board.size - 1) + "\n"
    print(result)


def play_game():
    print(len(HEURISTIC))
    game_board = None
    while True:
        size = int(input("Enter the board size : "))
        if size >= 5:
            game_board = Board(size, 123, 123)
            print_board(game_board)
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

        if game_board.check_valid_move((row, col)):
            game_board.board[row][col] = 'o'
            print()
            print_board(game_board)
        else:
            print("Invalid move! Please select an empty cell.")
            continue

        if game_board.check_status(game_board.board):
            break

        print("Bot is thinking....")
        print()

        make_move(game_board, 'x')
        print_board(game_board)

        if game_board.check_status(game_board.board):
            break


if __name__ == "__main__":
    while True:
        play_game()
        print("To quit, press q. To play again, press any key.")
        if input() == 'q':
            break
