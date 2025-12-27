"""
Connect4（四子棋）MuZero 多Actor并行采集配置

只需添加 n_actors 参数即可启用多Actor模式！
"""

from easydict import EasyDict

# ==============================================================
# 用户常用配置区域
# ==============================================================

# 多Actor配置
n_actors = 4                    # Actor数量
envs_per_actor = 8              # 每个Actor管理的环境数量

# 总环境数
collector_env_num = n_actors * envs_per_actor  # 32
n_episode = collector_env_num
evaluator_env_num = 5

# MCTS配置
num_simulations = 50

# 训练配置
update_per_collect = 50
reanalyze_ratio = 0.
batch_size = 256
max_env_step = int(5e5)

# ==============================================================
# 配置结束
# ==============================================================

connect4_muzero_config = dict(
    exp_name=f'data_muzero/connect4_muzero_multi_actor_sp-mode_actors{n_actors}_envs{envs_per_actor}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        bot_action_type='rule',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 6, 7),  # Connect4棋盘: 6行7列
            action_space_size=7,          # 7列可落子
            image_channel=3,
            num_res_blocks=1,
            num_channels=64,
            reward_support_range=(-300., 301., 1.),
            value_support_range=(-300., 301., 1.),
        ),
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=int(6 * 7),   # 最大步数
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        td_steps=int(6 * 7),
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        
        # ============================================
        # 多Actor配置
        # ============================================
        n_actors=n_actors,
        envs_per_actor=envs_per_actor,
    ),
)
connect4_muzero_config = EasyDict(connect4_muzero_config)
main_config = connect4_muzero_config

connect4_muzero_create_config = dict(
    env=dict(
        type='connect4',
        import_names=['zoo.board_games.connect4.envs.connect4_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
connect4_muzero_create_config = EasyDict(connect4_muzero_create_config)
create_config = connect4_muzero_create_config


if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

