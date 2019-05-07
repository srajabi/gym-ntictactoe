from gym.envs.registration import register
from gym_ntictactoe.envs.n_tictactoe import Player

register(
    id='tictactoe-v0',
    entry_point='gym_ntictactoe.envs:NTicTacToeEnv',
    kwargs={'order': 3, 'dimensions': 2}
)

register(
    id='tictactoe3d-v0',
    entry_point='gym_ntictactoe.envs:NTicTacToeEnv',
    kwargs={'order': 3, 'dimensions': 3}
)

register(
    id='tictactoe4d-v0',
    entry_point='gym_ntictactoe.envs:NTicTacToeEnv',
    kwargs={'order': 3, 'dimensions': 4}
)
