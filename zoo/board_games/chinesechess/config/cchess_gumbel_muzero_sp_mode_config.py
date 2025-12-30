"""
中国象棋 Gumbel MuZero 自对弈模式配置

Gumbel MuZero 特点：
- 使用 Gumbel-Top-k 技巧进行动作采样，搜索更高效
- 使用 improved policy 作为训练目标
- 适合大动作空间（如中国象棋的 2238 种动作）

重构版本特性：
- 动作空间：2238（压缩后的合法移动）
- 观察空间：(56, 10, 9) = 14层棋子 * 4历史帧，己方优先编码
- 去除颜色层，固定视角
"""

from easydict import EasyDict
from zoo.board_games.chinesechess.envs.action_mapping import ACTION_SPACE_SIZE

# ==============================================================
# 常用配置参数
# ==============================================================

# 多GPU配置
use_multi_gpu = False  # 是否开启多GPU训练
gpu_num = 1

# 环境配置
collector_env_num = 8
evaluator_env_num = 3
n_episode = 8

# MCTS 配置
num_simulations = 50  # 模拟次数，可根据需要调整

# 训练配置
batch_size = 256
update_per_collect = 50
reanalyze_ratio = 0.0
max_env_step = int(1e7)

# ==============================================================
# 配置结束
# ==============================================================

cchess_gumbel_muzero_config = dict(
    exp_name=f'data_gumbel_muzero/cchess_gumbel_muzero_sp-mode_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        # 渲染配置（可选）
        # render_mode='html',
        # replay_path='./replay',
    ),
    policy=dict(
        model=dict(
            # 观察空间：14层棋子 * 4历史帧 = 56层（去除颜色层）
            observation_shape=(56, 10, 9),
            # 动作空间：压缩后的合法移动
            action_space_size=ACTION_SPACE_SIZE,  # 2238
            image_channel=56,
            # 网络结构
            num_res_blocks=9,
            num_channels=128,
            # 支持范围（棋类游戏奖励为 -1/0/1）
            reward_support_range=(-2., 3., 1.),
            value_support_range=(-2., 3., 1.),
        ),
        # 预训练模型路径（可选）
        model_path=None,
        cuda=True,
        multi_gpu=use_multi_gpu,
        env_type='board_games',
        action_type='varied_action_space',
        # MCTS 树搜索使用 C++ 实现
        mcts_ctree=True,
        # 游戏片段长度
        game_segment_length=100,
        # 训练参数
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0003,
        grad_clip_value=0.5,
        # MCTS 参数
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # ============================================
        # Gumbel MuZero 特有配置
        # ============================================
        max_num_considered_actions=16,  # Gumbel 采样考虑的最大动作数
        gumbel_algo=True,
        # TD 学习步数（棋类游戏设置较大以确保 value target 是最终结果）
        td_steps=30,
        num_unroll_steps=5,
        # 棋类游戏使用 discount_factor=1
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(500),
        replay_buffer_size=int(2e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cchess_gumbel_muzero_config = EasyDict(cchess_gumbel_muzero_config)
main_config = cchess_gumbel_muzero_config

cchess_gumbel_muzero_create_config = dict(
    env=dict(
        type='cchess',
        import_names=['zoo.board_games.chinesechess.envs.cchess_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='gumbel_muzero',
        import_names=['lzero.policy.gumbel_muzero'],
    ),
)

cchess_gumbel_muzero_create_config = EasyDict(cchess_gumbel_muzero_create_config)
create_config = cchess_gumbel_muzero_create_config


if __name__ == "__main__":
    from lzero.entry import train_muzero
    
    train_muzero(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step
    )
