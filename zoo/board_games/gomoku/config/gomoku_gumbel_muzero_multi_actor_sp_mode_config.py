"""
五子棋 Gumbel MuZero 多Actor并行采集配置

Gumbel MuZero特点:
- 使用Gumbel采样进行动作选择
- 更高效的MCTS搜索
- 支持improved policy输出

使用MultiActorMuZeroCollector实现多Actor并行采集，提升GPU利用率。
"""

from easydict import EasyDict

# ==============================================================
# 用户常用配置区域
# ==============================================================

# 多Actor配置（核心新增配置）
n_actors = 4                    # Actor数量，根据CPU核心数调整
envs_per_actor = 8              # 每个Actor管理的环境数量

# 总环境数 = n_actors * envs_per_actor
collector_env_num = n_actors * envs_per_actor  # 32
n_episode = collector_env_num   # 每次collect收集的episode数

# 评估配置
evaluator_env_num = 5

# MCTS配置
num_simulations = 50

# 训练配置
update_per_collect = 50
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.

# 棋盘配置
board_size = 6  # 默认15，测试用6
bot_action_type = 'v0'  # options={'v0', 'v1'}
prob_random_action_in_bot = 0.5

# ==============================================================
# 配置结束
# ==============================================================

gomoku_gumbel_muzero_multi_actor_config = dict(
    exp_name=f'data_muzero/gomoku_gumbel_muzero_multi_actor_sp-mode_actors{n_actors}_envs{envs_per_actor}_rand{prob_random_action_in_bot}_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        board_size=board_size,
        battle_mode='self_play_mode',
        bot_action_type=bot_action_type,
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, board_size, board_size),
            action_space_size=int(board_size * board_size),
            image_channel=3,
            num_res_blocks=1,
            num_channels=32,
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
        ),
        # 模型路径，None表示从头训练
        model_path=None,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=int(board_size * board_size),  # self_play_mode
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        
        # ============================================
        # Gumbel MuZero 特有配置
        # ============================================
        max_num_considered_actions=6,       # Gumbel采样考虑的最大动作数
        gumbel_algo=True,                   # 标记使用Gumbel算法
        
        # 棋盘游戏设置大td_steps确保value target是最终结果
        td_steps=int(board_size * board_size),  # self_play_mode
        # 棋盘游戏discount_factor=1
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        
        # ============================================
        # 多Actor配置（新增）
        # ============================================
        n_actors=n_actors,                  # Actor数量
        envs_per_actor=envs_per_actor,      # 每个Actor的环境数
        use_multi_actor=True,               # 启用多Actor模式
    ),
)
gomoku_gumbel_muzero_multi_actor_config = EasyDict(gomoku_gumbel_muzero_multi_actor_config)
main_config = gomoku_gumbel_muzero_multi_actor_config

gomoku_gumbel_muzero_multi_actor_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='gumbel_muzero',
        import_names=['lzero.policy.gumbel_muzero'],
    ),
    # 使用多Actor收集器
    collector=dict(
        type='multi_actor_muzero',
        import_names=['lzero.worker.muzero_collector_multi_actor'],
    )
)
gomoku_gumbel_muzero_multi_actor_create_config = EasyDict(gomoku_gumbel_muzero_multi_actor_create_config)
create_config = gomoku_gumbel_muzero_multi_actor_create_config


if __name__ == "__main__":
    # 使用专门的多Actor训练入口
    from zoo.board_games.gomoku.entry.train_muzero_multi_actor import train_muzero_multi_actor
    train_muzero_multi_actor(
        [main_config, create_config], 
        seed=0, 
        model_path=main_config.policy.model_path, 
        max_env_step=max_env_step
    )

