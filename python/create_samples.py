# create 10k mate in 1 games for white

import chess
import numpy as np

max_wins = 10000
won_games = 0
game_num = 0

file = open('mio.txt', 'w')
while 1:
    # start with a new game
    board = chess.Board()
    moves = board.legal_moves
    found = None
    
    # run the game till completion
    game_num += 1
    print('Game', game_num, end=' ')
    while 2:
        # do random moves
        move = np.random.choice(list(moves))
        board.push(move)
        
        # check is the match is already over
        result = board.result(claim_draw=True)
        if result != '*':
            break
        # check the following moves for a mate
        moves = board.legal_moves
        for move in moves:
            board.push(move)
            result = board.result(claim_draw=True)
            board.pop()
            
            if result=='1-0':
                # found white mating
                if found:
                    found = None
                    break
                found = move
                
        # should have only 1 solution
        if found:
            won_games += 1
            print(found, won_games, end='')
            file.write(board.fen()+'='+found.uci())
            file.write('\n')
            file.flush()
            break
            
    print()
    if won_games>=max_wins:
        break
        
file.close()
