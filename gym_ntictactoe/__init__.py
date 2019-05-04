from gym.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='gym_ntictactoe.envs:NTicTacToeEnv',
    kwargs={'order': 3, 'dimensions': 2}
)

register(
    id='tictactoe3d-v0',
    entry_point='gym_ntictactoe:NTicTacToeEnv',
    kwargs={'order': 3, 'dimensions': 3}
)
