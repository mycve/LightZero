from easydict import EasyDict
from zoo.board_games.chinesechess.envs.action_mapping import ACTION_SPACE_SIZE

# ==============================================================
# 最常修改的配置参数
# ==============================================================
# 多GPU配置
use_multi_gpu = True  # 开启多GPU训练
gpu_num = 8  # 使用的GPU数量，根据实际情况修改
batch_size = 128

collector_env_num = 128
n_episode = 128
evaluator_env_num = 10
num_simulations = 20  # 增加到 400 以提升搜索质量,目前简单测试时，先设置为20
update_per_collect = 10
reanalyze_ratio = 0.0  # 利用MuZero重分析优势，提升样本利用率
max_env_step = int(1e8)  # 中国象棋需要更多训练步数
max_episode_steps = 200  # 最大回合数
# ==============================================================
# 配置参数结束
# ==============================================================

cchess_muzero_config = dict(
    exp_name=f'data_muzero/cchess_self-play-mode_seed0',
    env=dict(
        battle_mode='self_play_mode',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=True, ),
        # 游戏规则
        max_episode_steps=max_episode_steps,
        draw_as_loss=True,  # 和棋判双方都输
        # UCI引擎配置（可选，用于eval_mode评估）
        # uci_engine_path='pikafish',
        # engine_depth=10,
    ),
    policy=dict(
        model=dict(
            model_type='conv',
            # 14层(7己+7敌) * 4历史 = 56层（去除颜色层）
            observation_shape=(56, 10, 9),
            action_space_size=ACTION_SPACE_SIZE,  # 2238 压缩后的动作空间
            image_channel=56,
            num_res_blocks=9,
            num_channels=128,
            reward_support_range=(-2., 3., 1.),
            value_support_range=(-2., 3., 1.),
        ),
        cuda=True,
        multi_gpu=use_multi_gpu,  # 开启多GPU数据并行
        env_type='board_games',
        action_type='varied_action_space',
        mcts_ctree=True,
        game_segment_length=100,  # 中国象棋平均步数较多
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0003,  # 从0.003降到0.0003，避免训练震荡
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        num_unroll_steps=5,  # MuZero展开步数
        td_steps=30,  # TD学习步数，需要满足：game_segment_length > td_steps + num_unroll_steps
        discount_factor=1,  # 棋类游戏使用 1
        n_episode=n_episode,
        eval_freq=int(200),
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
    from ding.utils import DDPContext
    from lzero.entry import train_muzero
    from lzero.config.utils import lz_to_ddp_config

    # ==============================================================
    # 兼容 Ding 日志聚合：在调用 learner 的 hook 之前，把 log_buffer
    # 里的 numpy.ndarray 转成 Python 标量或 list，避免
    # "invalid type in reduce: <class 'numpy.ndarray'>"。
    # 只改 BaseLearner.call_hook，不动框架其他逻辑。
    # ==============================================================
    import numpy as np
    from ding.worker import BaseLearner

    def _sanitize_log_buffer_for_ndarray(data):
        if isinstance(data, dict):
            return {k: _sanitize_log_buffer_for_ndarray(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_sanitize_log_buffer_for_ndarray(v) for v in data]
        elif isinstance(data, np.ndarray):
            # 标量数组 -> 标量；向量/矩阵 -> Python list
            if data.size == 1:
                return data.item()
            else:
                return data.tolist()
        else:
            return data

    _orig_call_hook = BaseLearner.call_hook

    def _patched_call_hook(self, place: str):
        # 只在 after_iter 前做一次清洗，其他 hook 保持原样
        if place == 'after_iter' and getattr(self, 'log_buffer', None) is not None:
            try:
                self.log_buffer = _sanitize_log_buffer_for_ndarray(self.log_buffer)
            except Exception:
                # 清洗失败时不影响训练流程
                pass
        return _orig_call_hook(self, place)

    BaseLearner.call_hook = _patched_call_hook

    seed = 0
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
