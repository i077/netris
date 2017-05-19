#!/usr/bin/env python2
#-*- coding: utf-8 -*-

# NOTE FOR WINDOWS USERS:
# You can download a "exefied" version of this game at:
# http://kch42.de/progs/tetris_py_exefied.zip
# If a DLL is missing or something like this, write an E-Mail (kevin@kch42.de)
# or leave a comment on this gist.

# Very simple tetris implementation
#
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop
#
# Have fun!

# Copyright (c) 2010 "Kevin Chabowski"<kevin@kch42.de>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from random import randrange as rand
import pygame, sys
from network import *
from copy import deepcopy
from random import randrange, randint

# The configuration
cell_size =	18
cols =		12
rows =		22
maxfps = 	1000

keys_y = 100
keys_x = 100
keys_spacing = 20

colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35),
(134, 134, 134),
(0, 255, 255)
]

# Define the shapes of the single parts
tetris_shapes = [
    [ # T pieces
        [[0, 0, 0],
         [1, 1, 1],
         [0, 1, 0]],
        
        [[0, 1, 0],
         [1, 1, 0],
         [0, 1, 0]],

        [[0, 1, 0],
         [1, 1, 1],
         [0, 0, 0]],

        [[0, 1, 0],
         [0, 1, 1],
         [0, 1, 0]],
    ],

    [ # S pieces
        [[0, 0, 0],
         [0, 2, 2],
         [2, 2, 0]],
        
        [[0, 2, 0],
         [0, 2, 2],
         [0, 0, 2]]
    ],

    [ # Z pieces
        [[0, 0, 0],
         [3, 3, 0],
         [0, 3, 3]],
        
        [[0, 0, 3],
         [0, 3, 3],
         [0, 3, 0]]
    ],

    [ # J pieces
        [[0, 0, 0],
         [4, 4, 4],
         [0, 0, 4]],

        [[0, 4, 0],
         [0, 4, 0],
         [4, 4, 0]],

        [[4, 0, 0],
         [4, 4, 4],
         [0, 0, 0]],

        [[0, 4, 4],
         [0, 4, 0],
         [0, 4, 0]]

    ],

    [ # L pieces
        [[0, 0, 0],
         [5, 5, 5],
         [5, 0, 0]],

        [[5, 5, 0],
         [0, 5, 0],
         [0, 5, 0]],

        [[0, 0, 5],
         [5, 5, 5],
         [0, 0, 0]],

        [[0, 5, 0],
         [0, 5, 0],
         [0, 5, 5]]

    ],

    [ # I pieces
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [6, 6, 6, 6],
         [0, 0, 0, 0]],

        [[0, 0, 6, 0],
         [0, 0, 6, 0],
         [0, 0, 6, 0],
         [0, 0, 6, 0]],
    ],

    [ # O piece
        [[7, 7],
         [7, 7]]
    ]
]

def go_to_next_index(list, index, dir):
    new_index = index
    if dir == 1:
        new_index = 0 if index == len(list) - 1 else index + 1
    elif dir == -1:
        new_index = len(list) - 1 if index == 0 else index - 1
    return new_index

def rotate_clockwise(shape): # This shouldn't get used
    return [ [ shape[y][x]
            for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1) ]

def rotate_counterclockwise(shape): # This shouldn't get used
    for i in range(3):
        shape = rotate_clockwise(shape)
    return shape

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[ cy + off_y ][ cx + off_x ]:
                    return True
            except IndexError:
                return True
    return False

def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board

def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            try:
                mat1[cy+off_y-1	][cx+off_x] += val
            except IndexError:
                pass
    return mat1

def new_board():
    board = [ [ 0 for x in range(cols) ]
            for y in range(rows) ]
    board += [[ 1 for x in range(cols)]]
    return board

class TetrisApp(object):
    def __init__(self):
        pygame.init()
        # pygame.key.set_repeat(250,25)
        self.width = cell_size*(cols+6)
        self.height = cell_size*rows
        self.rlim = cell_size*cols
        self.bground_grid = [[ 8 if x%2==y%2 and y > 1 else 0 for x in range(cols)] for y in range(rows)]

        self.default_font =  pygame.font.Font(
            pygame.font.get_default_font(), 12)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
                                                     # mouse movement
                                                     # events, so we
                                                     # block them.
        self.next_stone_index = rand(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.next_stone_index][0]
        self.next_stone_variation_index = 0

        # Training stuff
        self.training_data = []
        self.scores = []
        self.accepted_scores = []
        self.prev_observation = []
        self.number_of_games = 0

        self.init_game()

    def new_stone(self):
        self.stone = self.next_stone[:]
        self.stone_index = self.next_stone_index
        self.stone_variation_index = self.next_stone_variation_index
        self.next_stone_index = rand(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.next_stone_index][0]
        self.next_stone_variation_index = 0
        # self.stone_x = int(cols / 2 - len(self.stone[0])/2)
        self.stone_x = 5
        self.stone_y = 1
        if self.stone_index == 5:
            self.stone_x = 4
            self.stone_y = 0
        elif self.stone_index == 6:
            self.stone_y = 2

        if check_collision(self.board,
                           self.stone,
                           (self.stone_x, self.stone_y)) or self.check_top_rows():
            self.number_of_games += 1
            self.gameover = True
            on_gameover(self)

    def check_top_rows(self):
        if rows > 20 :
            for i in range(1,cols-1):
                if self.board[0][i] or self.board[1][i]:
                    return True
        return False

    def init_game(self):
        self.board = new_board()
        self.new_stone()
        self.game_memory = []
        self.level = 1
        self.score = 0
        self.lines = 0
        max_tick = int(1000 * (30 / maxfps))
        pygame.time.set_timer(pygame.USEREVENT+1, max_tick)

    def disp_msg(self, msg, topleft):
        x,y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255,255,255),
                    (0,0,0)),
                (x,y))
            y+=14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image =  self.default_font.render(line, False,
                (255,255,255), (0,0,0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
              self.width // 2-msgim_center_x,
              self.height // 2-msgim_center_y+i*22))

    def draw_matrix(self, matrix, offset):
        off_x, off_y  = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    #print(val)
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x+x) *
                              cell_size,
                            (off_y+y) *
                              cell_size,
                            cell_size,
                            cell_size),0)


    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level*10:
            self.level += 1
            #newdelay = 1000-50*(self.level-1)
            #newdelay = 100 if newdelay < 100 else newdelay
            #pygame.time.set_timer(pygame.USEREVENT+1, newdelay)

    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            '''
            if new_x < 0:
                    #new_x = 0
                empty = True
                for i in range(len(tetris_shapes[self.stone_index]) -1):
                    col = -(1 + new_x)
                    if tetris_shapes[self.stone_index][self.stone_variation_index][i][col] != 0:
                        empty = False
                        break
                if not empty:
                    new_x = self.stone_x
            #if new_x > cols - len(self.stone[0]):
                    #new_x = cols - len(self.stone[0])
            '''
            if not check_collision(self.board,
                                   self.stone,
                                   (new_x, self.stone_y)):
                self.stone_x = new_x
    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self, manual):
        if not self.gameover and not self.paused:
            self.score += 1 if manual else 0
            self.stone_y += 1
            if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                  self.board,
                  self.stone,
                  (self.stone_x, self.stone_y))
                self.new_stone()
                self.cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(
                              self.board, i)
                            self.cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(self.cleared_rows)
                return True
        return False

    def insta_drop(self):
        if not self.gameover and not self.paused:
            while(not self.drop(True)):
                pass

    def rotate_stone(self, dir):
        if not self.gameover and not self.paused:
            # new_stone = rotate_clockwise(self.stone)
            new_stone_variation_index = go_to_next_index(tetris_shapes[self.stone_index], self.stone_variation_index, dir)
            if not check_collision(self.board,
                                   tetris_shapes[self.stone_index][new_stone_variation_index],
                                   (self.stone_x, self.stone_y)):
                self.stone = tetris_shapes[self.stone_index][new_stone_variation_index]
                self.stone_variation_index = new_stone_variation_index

    def rotate_stone_ccw(self): # This is never used
        if not self.gameover and not self.paused:
            # new_stone = rotate_counterclockwise(self.stone)
            new_stone = go_to_next_index(tetris_shapes[self.stone_index], self.stone_variation_index, -1)
            if not check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def prep_board(self, gameboard, piece):
        new_piece = deepcopy(piece)
        new_gameboard = deepcopy(gameboard)
        for i, row in enumerate(new_piece):
            for j, cell in enumerate(row):
                if cell:
                    new_piece[i][j] = 10
        new_gameboard = join_matrixes(new_gameboard, new_piece, (self.stone_x, self.stone_y))
        return new_gameboard

    def prep_current_board(self):
        return self.prep_board(self.board, self.stone)

    def one_hot_to_inputs(self, one_hot):
        key = ''
        keys = ['LEFT', 'RIGHT', 'd', 'f']
        if 1 not in one_hot:
            return
        for i, k in enumerate(one_hot[:-1]):
            if k:
                key = keys[i]
                break
        if key:
            self.key_actions[key]()

    def run(self):
        self.key_actions = {
            'ESCAPE':	self.quit,
            'LEFT':		lambda:self.move(-1),
            'RIGHT':	lambda:self.move(+1),
            'DOWN':		lambda:self.drop(True),
            'UP':		lambda:self.rotate_stone(1),
            'p':		self.toggle_pause,
            'SPACE':	self.start_game,
            'RETURN':	self.insta_drop,
            'd':        lambda:self.rotate_stone(-1),
            'f':        lambda:self.rotate_stone(1)
        }

        self.gameover = False
        self.paused = False

        dont_burn_my_cpu = pygame.time.Clock()
        while 1:
            self.random_training()
            create_training_data(self)
            # print(readboard(self.prep_board(self.board, self.stone)))
            for i in range(rows):
                self.board[i][0] = 9
                self.board[i][cols - 1] = 9
            self.screen.fill((0,0,0))
            if self.gameover:
                self.center_msg("""Game Over!\nYour score: %d
Press space to continue""" % self.score)
            else:
                if self.paused:
                    self.center_msg("Paused")
                else:
                    pygame.draw.line(self.screen,
                        (255,255,255),
                        (self.rlim+1, 0),
                        (self.rlim+1, self.height-1))
                    self.disp_msg("Next:", (
                        self.rlim+cell_size,
                        2))
                    self.disp_msg("Score: %d\n\nLevel: %d\
\nLines: %d" % (self.score, self.level, self.lines),
                        (self.rlim+cell_size, cell_size*5))
                    self.draw_matrix(self.bground_grid, (0,0))
                    self.draw_matrix(self.board, (0,0))
                    self.draw_matrix(self.stone,
                        (self.stone_x, self.stone_y))
                    self.draw_matrix(self.next_stone,
                        (cols+1,2))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT+1:
                    self.drop(False)
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    for key in self.key_actions:
                        if event.key == eval("pygame.K_"
                        +key):
                            self.key_actions[key]()

            dont_burn_my_cpu.tick(maxfps)
    
    # Random training
    def random_training(self):
        inputindex = randrange(12)
        self.action = gen_onehot(inputindex)
        self.one_hot_to_inputs(self.action)

if __name__ == '__main__':
    App = TetrisApp()
    App.run()
