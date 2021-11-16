from state import State, State_2, UltimateTTT_Move
import time
from importlib import import_module
#Addition Code begins
import numpy as np
def evaluate_global(cur_state:State)->int:
    """Return evaluate score of a state"""
    score = cur_state.game_result(cur_state.global_cells.reshape(3, 3))
    if (score != None): return 100 * score
    return 0

def get_valid_moves_test(cur_state, player, previous_move):
        if previous_move != None:
            cur_state.index_local_board = previous_move.x * 3 + previous_move.y
        else: 
            temp_blocks = np.zeros((3, 3))
            indices = np.where(temp_blocks == 0)
            ret = []
            for i in range(9):
                ret += [UltimateTTT_Move(i, index[0], index[1], player)
                        for index in list(zip(indices[0], indices[1]))
                    ]
            return ret
            
        local_board = cur_state.blocks[cur_state.index_local_board]
        indices = np.where(local_board == 0)
        
        if(len(indices[0]) != 0):
            cur_state.free_move = False
            return [UltimateTTT_Move(cur_state.index_local_board, index[0], 
                                     index[1], player)
                    for index in list(zip(indices[0], indices[1]))
                ]
        # chosen board is full      
        cur_state.free_move = True        
        ret = []
        for i in range(9):
            if not np.all(cur_state.blocks[i] != 0):
                indices = np.where(cur_state.blocks[i] == 0)
                ret += [UltimateTTT_Move(i, index[0], index[1], player)
                        for index in list(zip(indices[0], indices[1]))
                    ]
        return ret

def act_move_test(cur_state:State, move:UltimateTTT_Move):
    """Move without affect to real game"""
    cur_state.local_board = cur_state.blocks[move.index_local_board]
    cur_state.local_board[move.x, move.y] = move.value

    if cur_state.global_cells[move.index_local_board] == 0: # not 'X' or 'O'
        if cur_state.game_result(cur_state.local_board):
            cur_state.global_cells[move.index_local_board] = move.value

def minimax(cur_state, depth, player, previous_move)->int:
    """Return a score that state can lead to"""
    if depth == 2: return 0

    if cur_state.game_over == True:
        return evaluate_global(cur_state)

    valid_moves = get_valid_moves_test(cur_state, player, previous_move)
    print(valid_moves)
    print("************************************************************************")
    if player == 1:
        bestVal = -np.inf
        for move in valid_moves:
            copy_state = State(cur_state)
            act_move_test(copy_state, move)
            pre_move = move
            value = minimax(copy_state, depth + 1, -1, pre_move)
            bestVal = max(bestVal, value)
        return bestVal

    else:
        bestVal = np.inf
        for move in valid_moves:
            copy_state = State(cur_state)
            act_move_test(copy_state, move)
            pre_move = move
            value = minimax(copy_state, depth + 1, 1, pre_move)
            bestVal = min(bestVal, value)
        return bestVal

def select_move(cur_state, remain_time):
    player = cur_state.player_to_move
    pre_move = cur_state.previous_move
    valid_moves = get_valid_moves_test(cur_state, -1, pre_move )
    best_move = valid_moves[0]
    for move in valid_moves:

        copy_state_1 = State(cur_state)
        act_move_test(copy_state_1, move)

        copy_state_2 = State(cur_state)
        act_move_test(copy_state_2, best_move)
        
        if (minimax(copy_state_1, 0, 1, pre_move) < minimax(copy_state_2, 0, 1, pre_move)):
            best_move = move
    return best_move

def heuristic(cur_state:State):
    """evaluate the game is not over"""
    """O is minimal"""
    score = 0
    board = cur_state.global_cells.reshape(3, 3)
    for i in range(3):
        for j in range(3):
            if board[i][j] == -1:
                score -= 5
    return score

#Addition Code ends
  
def main(player_X, player_O, rule = 1):
    dict_player = {1: 'X', -1: 'O'}
    if rule == 1:
        cur_state = State()
    else:
        cur_state = State_2()
    turn = 1    

    limit = 81
    remain_time_X = 120
    remain_time_O = 120
    
    player_1 = import_module(player_X)
    player_2 = import_module(player_O)
    
    
    while turn <= limit:
        print("turn:", turn, end='\n\n')
        if cur_state.game_over:
            print("winner:", dict_player[cur_state.player_to_move * -1])
            break
        
        start_time = time.time()
        if cur_state.player_to_move == 1:
            new_move = player_1.select_move(cur_state, remain_time_X)
            elapsed_time = time.time() - start_time
            remain_time_X -= elapsed_time
        else:
            new_move = player_2.select_move(cur_state, remain_time_O)
            elapsed_time = time.time() - start_time
            remain_time_O -= elapsed_time
            
        if new_move == None:
            break
        
        if remain_time_X < 0 or remain_time_O < 0:
            print("out of time")
            print("winner:", dict_player[cur_state.player_to_move * -1])
            break
                
        if elapsed_time > 10.0:
            print("elapsed time:", elapsed_time)
            print("winner: ", dict_player[cur_state.player_to_move * -1])
            break
        
        cur_state.act_move(new_move)
        print(cur_state)
        
        turn += 1
        
    print("X:", cur_state.count_X)
    print("O:", cur_state.count_O)
main('random_agent', '_MSSV')

 
