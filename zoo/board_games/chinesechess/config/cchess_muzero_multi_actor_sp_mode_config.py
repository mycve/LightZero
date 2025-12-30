"""
中国象棋 MuZero 多Actor并行采集配置

使用 MultiActorMuZeroCollector 实现多Actor并行采集，提升GPU利用率。
"""

from easydict import EasyDict
from zoo.board_games.chinesechess.envs.action_mapping import ACTION_SPACE_SIZE

# ==============================================================
# 常用配置参数
# ==============================================================

use_multi_gpu = False
gpu_num = 1

# 多Actor配置
n_actors = 8
envs_per_actor = 16
collector_env_num = n_actors * envs_per_actor
n_episode = collector_env_num

evaluator_env_num = 3
num_simulations = 50
batch_size = 256
update_per_collect = 50
reanalyze_ratio = 0.0
max_env_step = int(1e7)
max_episode_steps = 200

# ==============================================================
# 配置结束
# ==============================================================

cchess_muzero_multi_actor_config = dict(
    exp_name=f'data_muzero/cchess_muzero_multi_actor_sp-mode_actors{n_actors}_envs{envs_per_actor}_ns{num_simulations}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        max_episode_steps=max_episode_steps,
        draw_as_loss=True,
    ),
    policy=dict(
        model=dict(
            model_type='conv',
            observation_shape=(56, 10, 9),
            action_space_size=ACTION_SPACE_SIZE,
            image_channel=56,
            num_res_blocks=9,
            num_channels=128,
            reward_support_range=(-2., 3., 1.),
            value_support_range=(-2., 3., 1.),
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
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        td_steps=30,
        num_unroll_steps=5,
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(500),
        replay_buffer_size=int(2e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # 多Actor配置
        n_actors=n_actors,
        envs_per_actor=envs_per_actor,
        use_multi_actor=True,
    ),
)

cchess_muzero_multi_actor_config = EasyDict(cchess_muzero_multi_actor_config)
main_config = cchess_muzero_multi_actor_config

cchess_muzero_multi_actor_create_config = dict(
    env=dict(
        type='cchess',
        import_names=['zoo.board_games.chinesechess.envs.cchess_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='multi_actor_muzero',
        import_names=['lzero.worker.muzero_collector_multi_actor'],
    )
)

cchess_muzero_multi_actor_create_config = EasyDict(cchess_muzero_multi_actor_create_config)
create_config = cchess_muzero_multi_actor_create_config


if __name__ == "__main__":
    from zoo.board_games.gomoku.entry.train_muzero_multi_actor import train_muzero_multi_actor
    
    train_muzero_multi_actor(
        [main_config, create_config],
        seed=0,
        model_path=main_config.policy.model_path,
        max_env_step=max_env_step
    )
