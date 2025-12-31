"""
中国象棋 MuZero 配置文件
支持 DDP 多卡训练

观察空间：(68, 10, 9) = 17层 × 4历史帧
- 14层棋子（己方7层 + 对方7层）
- 3层特征（重复计数、步数、限着计数）

启动方式：
    单卡: python zoo/board_games/chinesechess/config/cchess_muzero_config.py
    多卡: torchrun --nproc_per_node=8 zoo/board_games/chinesechess/config/cchess_muzero_config.py
"""

import sys
import os
# 自动添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easydict import EasyDict
from zoo.board_games.chinesechess.envs.action_mapping import ACTION_SPACE_SIZE

# ==============================================================
# 常用配置参数（用户可修改区域）
# ==============================================================
collector_env_num = 32
n_episode = 256
evaluator_env_num = 8
num_simulations = 50
update_per_collect = 50
reanalyze_ratio = 0.0
batch_size = 256
max_env_step = int(1e7)
max_episode_steps = 200
# ==============================================================
# 配置结束
# ==============================================================

cchess_muzero_config = dict(
    exp_name=f'data_muzero/cchess_muzero_ns{num_simulations}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=True),
        max_episode_steps=max_episode_steps,
    ),
    policy=dict(
        model=dict(
            model_type='conv',
            observation_shape=(68, 10, 9),  # 17层 × 4帧 = 68
            action_space_size=ACTION_SPACE_SIZE,
            image_channel=68,  # 与 observation_shape[0] 一致
            num_res_blocks=9,
            num_channels=128,
            reward_support_range=(-2., 3., 1.),
            value_support_range=(-2., 3., 1.),
        ),
        model_path=None,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        mcts_ctree=True,
        game_segment_length=100,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE: 棋盘游戏设置大的 td_steps 确保 value target 是最终结果
        td_steps=30,
        num_unroll_steps=5,
        # NOTE: 棋盘游戏设置 discount_factor=1
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(500),
        replay_buffer_size=int(2e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cchess_muzero_config = EasyDict(cchess_muzero_config)
main_config = cchess_muzero_config

cchess_muzero_create_config = dict(
    env=dict(
        type='cchess',
        import_names=['zoo.board_games.chinesechess.envs.cchess_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)

cchess_muzero_create_config = EasyDict(cchess_muzero_create_config)
create_config = cchess_muzero_create_config


if __name__ == "__main__":
    """
    单卡训练:
        python zoo/board_games/chinesechess/config/cchess_muzero_config.py
    
    多卡 DDP 训练:
        torchrun --nproc_per_node=8 zoo/board_games/chinesechess/config/cchess_muzero_config.py
    """
    from ding.utils import DDPContext
    from lzero.entry import train_muzero
    from lzero.config.utils import lz_to_ddp_config
    
    with DDPContext():
        # 自动根据 world_size 调整 batch_size, n_episode 等参数
        main_config = lz_to_ddp_config(main_config)
        train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
