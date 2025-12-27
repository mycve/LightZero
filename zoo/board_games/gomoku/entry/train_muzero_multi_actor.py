"""
多Actor并行采集训练入口

适用于所有MuZero系列算法：
- MuZero
- Gumbel MuZero  
- EfficientZero
- Sampled MuZero
等

核心优势:
- N个Actor并行运行，谁先ready谁先推理
- GPU不再空闲等待，连续推理
- 绕过单核CPU性能瓶颈
"""

import logging
import os
from functools import partial
from typing import Optional, Tuple

import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, log_buffer_run_time
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MultiActorMuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator


def random_collect(
    policy_cfg, 
    policy, 
    random_policy_class, 
    collector, 
    replay_buffer,
    env_fn=None,
    env_config=None,
):
    """
    随机采集用于预填充replay buffer
    """
    random_policy = random_policy_class(cfg=policy_cfg)
    # 对于多Actor collector，暂时使用简单方式
    collector._policy = random_policy.collect_mode
    
    collect_kwargs = {
        'temperature': 1.0,
        'epsilon': 1.0,  # 完全随机
    }
    new_data = collector.collect(
        train_iter=0, 
        policy_kwargs=collect_kwargs
    )
    replay_buffer.push_game_segments(new_data)
    
    # 恢复原策略
    collector._policy = policy.collect_mode


def calculate_update_per_collect(cfg, new_data):
    """计算每次collect后的更新次数"""
    if cfg.policy.update_per_collect is not None:
        return cfg.policy.update_per_collect
    
    # 默认根据数据量计算
    collected_transitions = sum(len(seg.reward_segment) for seg in new_data[0])
    return max(1, collected_transitions // cfg.policy.batch_size)


def train_muzero_multi_actor(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    多Actor并行采集训练入口
    
    Args:
        input_cfg: (main_config, create_config) 配置元组
        seed: 随机种子
        model: 预定义模型（可选）
        model_path: 预训练模型路径（可选）
        max_train_iter: 最大训练迭代次数
        max_env_step: 最大环境步数
        
    Returns:
        训练完成的策略
    """
    cfg, create_cfg = input_cfg
    
    # 验证算法类型
    supported_algos = [
        'efficientzero', 'muzero', 'muzero_context', 'muzero_rnn_full_obs',
        'sampled_efficientzero', 'sampled_muzero', 'gumbel_muzero', 'stochastic_muzero'
    ]
    assert create_cfg.policy.type in supported_algos, \
        f"train_muzero_multi_actor支持的算法: {supported_algos}"

    # 根据算法类型选择GameBuffer
    if create_cfg.policy.type in ['muzero', 'muzero_context', 'muzero_rnn_full_obs']:
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_muzero':
        from lzero.mcts import SampledMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'stochastic_muzero':
        from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer

    # 设备配置
    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    
    # 获取环境配置
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    
    # 创建评估环境（评估器仍使用原方式）
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    # wandb配置
    if cfg.policy.use_wandb:
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )

    # 创建策略
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # 加载预训练模型
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # 创建组件
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # 策略配置
    policy_config = cfg.policy
    batch_size = policy_config.batch_size
    
    # 创建replay buffer
    replay_buffer = GameBuffer(policy_config)
    
    # ==============================================================
    # 创建多Actor收集器（核心变更）
    # ==============================================================
    # 需要为MultiActorMuZeroCollector传递env_fn和env_config
    # 这样它可以为每个Actor创建独立的env_manager
    
    # 创建一个临时的collector_env用于获取action_space等信息
    temp_collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    temp_collector_env.seed(cfg.seed)
    temp_collector_env.launch()
    
    collector = Collector(
        env=temp_collector_env,           # 传入用于获取基本信息
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config,
        env_fn=env_fn,                    # 环境创建函数
        env_config=collector_env_cfg,     # 环境配置列表
    )
    
    evaluator = Evaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )

    # ==============================================================
    # 主训练循环
    # ==============================================================
    learner.call_hook('before_run')
    
    if policy_config.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    # 随机数据收集
    if cfg.policy.random_collect_episode_num > 0:
        logging.info(f"开始随机数据收集: {cfg.policy.random_collect_episode_num} episodes")
        random_collect(
            cfg.policy, policy, LightZeroRandomPolicy, 
            collector, replay_buffer,
            env_fn=env_fn, env_config=collector_env_cfg
        )

    # 评估随机策略
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

    while True:
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        log_buffer_run_time(learner.train_iter, replay_buffer, tb_logger)
        
        collect_kwargs = {}
        
        # 设置温度
        collect_kwargs['temperature'] = visit_count_temperature(
            policy_config.manual_temperature_decay,
            policy_config.fixed_temperature_value,
            policy_config.threshold_training_steps_for_final_temperature,
            trained_steps=learner.train_iter
        )

        # 设置epsilon
        if policy_config.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=policy_config.eps.start,
                end=policy_config.eps.end,
                decay=policy_config.eps.decay,
                type_=policy_config.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)
        else:
            collect_kwargs['epsilon'] = 0.0

        # 评估
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        # ==============================================================
        # 多Actor并行数据采集
        # ==============================================================
        logging.info(f"开始多Actor数据采集 (train_iter={learner.train_iter})")
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        # 计算更新次数
        update_per_collect = calculate_update_per_collect(cfg, new_data)

        # 保存数据到replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        # 训练
        for i in range(update_per_collect):
            if replay_buffer.get_num_of_transitions() > batch_size:
                train_data = replay_buffer.sample(batch_size, policy)
            else:
                logging.warning(
                    f'Replay buffer数据不足: batch_size={batch_size}, '
                    f'{replay_buffer}, 继续采集...'
                )
                break

            if policy_config.use_wandb:
                policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

            # 核心训练步骤
            log_vars = learner.train(train_data, collector.envstep)

            if cfg.policy.use_priority:
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # 检查终止条件
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logging.info(f"训练完成: envstep={collector.envstep}, train_iter={learner.train_iter}")
            break

    # 清理
    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    
    return policy


if __name__ == "__main__":
    # 示例：运行MuZero多Actor配置
    from zoo.board_games.gomoku.config.gomoku_muzero_multi_actor_sp_mode_config import main_config, create_config
    train_muzero_multi_actor(
        [main_config, create_config],
        seed=0,
        max_env_step=int(1e6)
    )

