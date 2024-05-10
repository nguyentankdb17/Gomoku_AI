import time
import numpy as np


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


MIN = float('-inf')
MAX = float('inf')

UTILITY = {
        '11111': 30000000,
        '22222': -30000000,
        '011110': 20000000,
        '022220': -20000000,
        '011112': 500000,
        '211110': 500000,
        '022221': -500000,
        '122220': -500000,
        '01110': 30000,
        '02220': -30000,
        '011010': 15000,
        '010110': 15000,
        '022020': -15000,
        '020220': -15000,
        '001112': 2000,
        '211100': 2000,
        '002221': -2000,
        '122200': -2000,
        '211010': 2000,
        '210110': 2000,
        '010112': 2000,
        '011012': 2000,
        '122020': -2000,
        '120220': -2000,
        '020221': -2000,
        '022021': -2000,
        '01100': 500,
        '00110': 500,
        '02200': -500,
        '00220': -500
    }



def btsConvert(board, player):
    '''
    convert board to col,row,and diag string arrays for easier interpreting
    '''
    newBoard = np.array(board)
    cList, rList, dList = [], [], []
    bdiag = [newBoard.diagonal(i) for i in range(newBoard.shape[1] - 5, -newBoard.shape[1] + 4, -1)]
    fdiag = [newBoard[::-1, :].diagonal(i) for i in range(newBoard.shape[1] - 5, -newBoard.shape[1] + 4, -1)]
    for dgd in bdiag:
        bdiagVals = ""
        for point in dgd:
            if point == ' ':
                bdiagVals += "0"
            elif point == player:
                bdiagVals += "1"
            else:
                bdiagVals += "2"
        dList.append(bdiagVals)
    for dgu in fdiag:
        fdiagVals = ""
        for point in dgu:
            if point == ' ':
                fdiagVals += "0"
            elif point == player:
                fdiagVals += "1"
            else:
                fdiagVals += "2"
        dList.append(fdiagVals)
    boardT = newBoard.copy().transpose()
    for col in boardT:
        colVals = ""
        for point in col:
            if point == ' ':
                colVals += "0"
            elif point == player:
                colVals += "1"
            else:
                colVals += "2"
        cList.append(colVals)
    for row in newBoard:
        rowVals = ""
        for point in row:
            if point == ' ':
                rowVals += "0"
            elif point == player:
                rowVals += "1"
            else:
                rowVals += "2"
        rList.append(rowVals)
    return dList + cList + rList


def points(board, player):  # evaluates
    '''
    assigns points for moves
    '''
    val = 0
    player1StrArr = btsConvert(board, player)
    for i in range(len(player1StrArr)):
        len1 = len(player1StrArr[i])
        for j in range(len1):
            n = j + 5
            if n <= len1:
                st = player1StrArr[i][j:n]
                if st in UTILITY:
                    val += UTILITY[st]
        for j in range(len1):
            n = j + 6
            if n <= len1:
                st = player1StrArr[i][j:n]
                if st in UTILITY:
                    val += UTILITY[st]
    return val


def getCoordsAround(board):
    '''
    get points around placed stones
    '''
    board_size = len(board)
    outTpl = np.nonzero(board)  # return tuple of all non zero points on board
    potentialValsCoord = {}
    for i in range(len(outTpl[0])):
        y = outTpl[0][i]
        x = outTpl[1][i]
        if y > 0:
            potentialValsCoord[(x, y - 1)] = 1
            if x > 0:
                potentialValsCoord[(x - 1, y - 1)] = 1
            if x < (board_size - 1):
                potentialValsCoord[(x + 1, y - 1)] = 1
        if x > 0:
            potentialValsCoord[(x - 1, y)] = 1
            if y < (board_size - 1):
                potentialValsCoord[(x - 1, y + 1)] = 1
        if y < (board_size - 1):
            potentialValsCoord[(x, y + 1)] = 1
            if x < (board_size - 1):
                potentialValsCoord[(x + 1, y + 1)] = 1
            if x > 0:
                potentialValsCoord[(x - 1, y + 1)] = 1
        if x < (board_size - 1):
            potentialValsCoord[(x + 1, y)] = 1
            if y > 0:
                potentialValsCoord[(x + 1, y - 1)] = 1
    finalValsX, finalValsY = [], []
    for key in potentialValsCoord:
        finalValsX.append(key[0])
        finalValsY.append(key[1])
    return finalValsX, finalValsY


def minimax(board, isMaximizer, depth, alpha, beta, player):  # alpha, beta
    '''
    Minimax with Alpha-Beta pruning (also computer is 1st Max in this implementation)
    alpha is best already explored option along path to root for maximizer(AI)
    beta is best already explored option along path to root for minimizer(AI Opponent)
    '''
    point = points(board, player)
    if depth == 2 or point >= 20000000 or point <= -20000000:
        return point
    size = len(board)
    if isMaximizer:  # THE MAXIMIZER
        best = MIN
        potentialValsX, potentialValsY = getCoordsAround(board)
        for i in range(len(potentialValsX)):
            if board[potentialValsY[i]][potentialValsX[i]] == ' ':
                board[potentialValsY[i]][potentialValsX[i]] = player
                score = minimax(board, False, depth + 1, alpha, beta, player)
                best = max(best, score)
                alpha = max(alpha, best)  # best AI Opponent move
                board[potentialValsY[i]][potentialValsX[i]] = ' '  # undoing
                if beta <= alpha:
                    break
        return best
    else:  # THE MINIMIZER
        best = MAX
        potentialValsX, potentialValsY = getCoordsAround(board)
        for i in range(len(potentialValsX)):
            if board[potentialValsY[i]][potentialValsX[i]] == ' ':
                opponent = 'x' if player == 'o' else 'o'
                board[potentialValsY[i]][potentialValsX[i]] = opponent
                score = minimax(board, True, depth + 1, alpha, beta, player)
                best = min(best, score)
                beta = min(beta, best)  # best AI Opponent move
                board[potentialValsY[i]][potentialValsX[i]] = ' '  # undoing
                if beta <= alpha:
                    break
        return best


def getRandomMove(board):
    '''
    For choosing random move when can't decide propogated to center
    '''
    print('siuu')
    boardSize = len(board)
    ctr = 0
    idx = boardSize // 2
    while ctr < (idx / 2):
        if board[idx + ctr][idx + ctr] == ' ':
            return idx + ctr, idx + ctr
        elif board[idx + ctr][idx - ctr] == ' ':
            return idx + ctr, idx - ctr
        elif board[idx + ctr][idx] == ' ':
            return idx + ctr, idx
        elif board[idx][idx + ctr] == ' ':
            return idx, idx + ctr
        elif board[idx][idx - ctr] == ' ':
            return idx, idx - ctr
        elif board[idx - ctr][idx] == ' ':
            return idx - ctr, idx
        elif board[idx - ctr][idx - ctr] == ' ':
            return idx - ctr, idx - ctr
        elif board[idx - ctr][idx + ctr] == ' ':
            return idx - ctr, idx + ctr
        ctr += 1
    for i in range(boardSize):
        for j in range(boardSize):
            if board[i][j] == ' ':
                return i, j


def get_move(board, player):
    mostPoints = float('-inf')
    alpha, beta = MIN, MAX
    bestMoveRow = bestMoveCol = -1
    boardSize = len(board)
    potentialValsX, potentialValsY = getCoordsAround(board)
    for i in range(len(potentialValsX)):
        if board[potentialValsY[i]][potentialValsX[i]] == ' ':
            board[potentialValsY[i]][potentialValsX[i]] = player
            movePoints = max(mostPoints, minimax(
                board, False, 1, alpha, beta, player))
            alpha = max(alpha, movePoints)
            board[potentialValsY[i]][potentialValsX[i]] = ' '
            if beta <= alpha:
                break
            if movePoints > mostPoints:
                bestMoveRow = potentialValsY[i]
                bestMoveCol = potentialValsX[i]
                mostPoints = movePoints
                if movePoints >= 20000000:
                    break
    if bestMoveRow == -1 or bestMoveCol == -1:
        bestMoveRow, bestMoveCol = getRandomMove(board)
    return [bestMoveRow, bestMoveCol]


def make_move(game_board, player):
    move = get_move(game_board.board, player)
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
