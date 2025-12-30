"""
中国象棋 EfficientZero 多进程并行采集配置

使用 MultiActorMuZeroCollector 实现多进程并行采集:
- 真正的多进程并行，绕过GIL限制
- 共享内存零拷贝，无序列化开销
- GPU持续推理，充分利用硬件性能

EfficientZero 特有:
- 自监督学习损失
- 更好的样本效率

支持多卡DDP训练:
- 设置 use_multi_gpu=True 和 gpu_num=8
"""

from easydict import EasyDict
from zoo.board_games.chinesechess.envs.action_mapping import ACTION_SPACE_SIZE

# ==============================================================
# 常用配置参数
# ==============================================================

# 多GPU配置（8卡DDP）
use_multi_gpu = True
gpu_num = 8

# 多Actor配置
n_actors = 4                     # 每张卡的Actor进程数量
envs_per_actor = 512              # 每个Actor管理的环境数量

collector_env_num = n_actors * envs_per_actor  # 64 per GPU
n_episode = collector_env_num

evaluator_env_num = 3
num_simulations = 50
batch_size = 512
update_per_collect = 50
reanalyze_ratio = 0.0
max_env_step = int(1e7)
max_episode_steps = 200

# ==============================================================
# 配置结束
# ==============================================================

cchess_efficientzero_config = dict(
    exp_name=f'data_efficientzero/cchess_efficientzero_{gpu_num}gpu_actors{n_actors}_envs{envs_per_actor}_ns{num_simulations}_seed0',
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
            # EfficientZero 特有
            self_supervised_learning_loss=True,
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
        # EfficientZero 特有
        ssl_loss_weight=2.0,
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

cchess_efficientzero_config = EasyDict(cchess_efficientzero_config)
main_config = cchess_efficientzero_config

cchess_efficientzero_create_config = dict(
    env=dict(
        type='cchess',
        import_names=['zoo.board_games.chinesechess.envs.cchess_env'],
    ),
    # Actor已经是独立进程，内部env_manager用base更高效
    env_manager=dict(type='base'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
)

cchess_efficientzero_create_config = EasyDict(cchess_efficientzero_create_config)
create_config = cchess_efficientzero_create_config


if __name__ == "__main__":
    """
    8卡DDP训练: torchrun --nproc_per_node=8 此文件
    """
    from ding.utils import DDPContext
    from lzero.entry import train_muzero
    from lzero.config.utils import lz_to_ddp_config
    
    import numpy as np
    from ding.worker import BaseLearner

    def _sanitize_log_buffer(data):
        if isinstance(data, dict):
            return {k: _sanitize_log_buffer(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_sanitize_log_buffer(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.item() if data.size == 1 else data.tolist()
        return data

    _orig_call_hook = BaseLearner.call_hook
    def _patched_call_hook(self, place: str):
        if place == 'after_iter' and getattr(self, 'log_buffer', None) is not None:
            try:
                self.log_buffer = _sanitize_log_buffer(self.log_buffer)
            except:
                pass
        return _orig_call_hook(self, place)
    BaseLearner.call_hook = _patched_call_hook

    seed = 0
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
