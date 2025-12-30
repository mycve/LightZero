"""
中国象棋 EfficientZero 自对弈模式配置

EfficientZero 相比 MuZero 的主要改进：
- 自监督学习损失（consistency loss）：利用环境观测的一致性进行额外监督
- 更好的样本效率：在相同数据量下能学到更好的表示

重构版本特性：
- 动作空间：2238（压缩后的合法移动）
- 观察空间：(56, 10, 9) = 14层棋子 * 4历史帧，己方优先编码
- 去除颜色层，固定视角
- 和棋判双方都输（鼓励进攻）
"""

from easydict import EasyDict
from zoo.board_games.chinesechess.envs.action_mapping import ACTION_SPACE_SIZE

# ==============================================================
# 常用配置参数
# ==============================================================

# 多GPU配置
use_multi_gpu = False
gpu_num = 1

# 环境配置
collector_env_num = 128
evaluator_env_num = 3
n_episode = 128

# MCTS 配置
num_simulations = 50

# 训练配置
batch_size = 256
update_per_collect = 50
reanalyze_ratio = 0.0
max_env_step = int(1e7)
max_episode_steps = 200  # 最大回合数

# ==============================================================
# 配置结束
# ==============================================================

cchess_efficientzero_config = dict(
    exp_name=f'data_efficientzero/cchess_efficientzero_sp-mode_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        # 游戏规则
        max_episode_steps=max_episode_steps,
        draw_as_loss=True,  # 和棋判双方都输
    ),
    policy=dict(
        model=dict(
            model_type='conv',  # 使用卷积模型
            # 观察空间：14层棋子 * 4历史帧 = 56层
            observation_shape=(56, 10, 9),
            # 动作空间：压缩后的合法移动
            action_space_size=ACTION_SPACE_SIZE,  # 2238
            image_channel=56,
            # 网络结构
            num_res_blocks=9,
            num_channels=128,
            # 支持范围
            reward_support_range=(-2., 3., 1.),
            value_support_range=(-2., 3., 1.),
            # ============================================
            # EfficientZero 特有配置
            # ============================================
            self_supervised_learning_loss=True,  # 启用自监督学习损失
        ),
        model_path=None,
        cuda=True,
        multi_gpu=use_multi_gpu,
        env_type='board_games',
        action_type='varied_action_space',
        mcts_ctree=True,
        game_segment_length=100,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # ============================================
        # EfficientZero 特有配置
        # ============================================
        ssl_loss_weight=2.0,  # 自监督学习损失权重
        # TD 学习步数
        td_steps=30,
        num_unroll_steps=5,
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(500),
        replay_buffer_size=int(2e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

cchess_efficientzero_config = EasyDict(cchess_efficientzero_config)
main_config = cchess_efficientzero_config

cchess_efficientzero_create_config = dict(
    env=dict(
        type='cchess',
        import_names=['zoo.board_games.chinesechess.envs.cchess_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
)

cchess_efficientzero_create_config = EasyDict(cchess_efficientzero_create_config)
create_config = cchess_efficientzero_create_config


if __name__ == "__main__":
    from lzero.entry import train_muzero
    
    train_muzero(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step
    )
