import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum


# From https://stackoverflow.com/a/39185702/2140744
def product_slices(n):
    for i in range(n):
        yield (
            np.index_exp[np.newaxis] * i +
            np.index_exp[:] +
            np.index_exp[np.newaxis] * (n - i - 1)
        )


def get_lines(n, k):
    """
    Returns:
        index (tuple):   an object suitable for advanced indexing to get all possible lines
        mask (ndarray):  a boolean mask to apply to the result of the above
    """
    fi = np.arange(k)
    bi = fi[::-1]
    ri = fi[:,None].repeat(k, axis=1)

    all_i = np.concatenate((fi[None], bi[None], ri), axis=0)

    # index which look up every possible line, some of which are not valid
    index = tuple(all_i[s] for s in product_slices(n))

    # We incrementally allow lines that start with some number of `ri`s, and an `fi`
    #  [0]  here means we chose fi for that index
    #  [2:] here means we chose an ri for that index
    mask = np.zeros((all_i.shape[0],)*n, dtype=np.bool)
    sl = np.index_exp[0]
    for i in range(n):
        mask[sl] = True
        sl = np.index_exp[2:] + sl

    return index, mask


class Player(Enum):
    X = -1
    O = 1
    Empty = 0


class NTicTacToe(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, order=3, dimensions=3):
        assert order >= 3, 'Board order must be at least 3'
        assert dimensions >= 2, 'Dimensions must be at least 2'

        self.dimensions = dimensions
        self.order = order

        self.board = np.zeros((order,) * dimensions).squeeze()

        self.action_space = spaces.Discrete(len(self.board) - 1)
        self.observation_space = spaces.Discrete(len(self.board) - 1)

        self.whos_move = Player.Empty

    def reset(self):
        self.board = np.zeros((self.order,) * self.dimensions).squeeze()

        return self.board.flatten()

    def step(self, action):
        assert (isinstance(action, tuple) and np.shape(action) == (2,), '''Action must be a tuple of 
                                            (player, board_pos) where board_pos is an int and player 
                                            is either -1 or 1 given shape {}'''.format(np.shape(action)))

        player, move = action

        assert np.shape(player) == () and np.shape(move) == (), '''Action must be tuple of (player, board_pos)
                                                                where player and board_pos are integers
                                                                given player shape {} and move shape {}
                                                                '''.format(np.shape(move), np.shape(move))

        assert self.observation_space.contains(move), '''Move is invalid, outside range of action space. 
                                                         Move given {}'''.format(move)

        if not isinstance(player, Player):
            player = Player(player)

        assert player == Player.X or player == Player.O, '''Player is invalid, must be -1 or 1, or use Player tuple. 
                                                            Given {}'''.format(player)

        if self.whos_move == Player.Empty:
            self.whos_move = player

        assert self.whos_move == player, '''Not player {}'s turn'''.format(player)

        x = move % 3
        tmp = move // 3

        y = tmp % 3
        tmp = tmp // 3

        z = tmp % 3
        tmp = tmp // 3

        assert tmp == 0, '''Converting flattened idx to composite idx failed, got {} remaining.'''.format(tmp)

        value = self.board[z, y, x]

        assert value == Player.Empty, '''Could not place at position (x,y,z) ({},{},{}) player {}
                                         already there.'''.format(x, y, z, value)

        self.board[z, y, x] = player

        indices, mask = get_lines(self.dimensions, self.order)
        all_lines = self.board[indices][mask]

        any_won = any(sum(r) in {self.order, self.order * -1} for r in all_lines)

        self.whos_move = Player(self.whos_move.value * -1)
        done = False
        info = {'turn': self.whos_move}

        if any_won:
            reward = 1
            done = True
        else:
            reward = 0

        return self.board.flatten(), reward, done, info


    def render(self, mode='human'):
        print(self.board)  # TODO make this better.

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_ranom, seed = seeding.np_random(seed)
        return [seed]
