"""
多Actor并行采集器 - 多进程 + 共享内存实现

架构设计:
┌─────────────────────────────────────────────────────────────────────┐
│  MultiActorMuZeroCollector (主进程)                                  │
│      ├── SharedObsBuffer (共享内存，存储obs)                         │
│      ├── SharedResponseBuffer (共享内存，存储推理结果)               │
│      ├── GPUInferenceServer (主进程线程，处理GPU推理)                │
│      │       └── policy.forward()                                   │
│      │                                                              │
│      └── ActorProcess × N (独立进程)                                 │
│              └── EnvManager (每个Actor管理自己的envs)                │
│                                                                     │
│  工作流程:                                                           │
│  1. 每个ActorProcess独立运行自己的envs                               │
│  2. Actor将obs写入共享内存，通过Queue发送元数据                       │
│  3. GPUInferenceServer从共享内存读取数据，执行推理                    │
│  4. 结果写入共享内存，Actor从共享内存读取结果                         │
│  5. 全程零拷贝，无序列化开销                                         │
└─────────────────────────────────────────────────────────────────────┘

优势:
- 真正的多进程并行，绕过GIL限制
- 共享内存零拷贝，无序列化开销
- GPU不再空闲等待，总有Actor ready可以推理
- 通用设计，支持所有MuZero系列算法
"""

import os
import time
import copy
import logging
import traceback
import multiprocessing as mp
from collections import deque, namedtuple
from typing import Optional, Any, List, Dict, Callable, Tuple, Union
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.multiprocessing as torch_mp
import wandb
from easydict import EasyDict
from ding.envs import BaseEnvManager, create_env_manager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size
from ding.worker.collector.base_serial_collector import ISerialCollector
from torch.nn import L1Loss

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation


# =============================================================================
# 共享内存缓冲区
# =============================================================================

class SharedObsBuffer:
    """
    共享内存观察缓冲区 - 实现进程间零拷贝数据传输
    
    设计要点:
    - 每个Actor预分配固定大小缓冲区
    - 使用 torch.Tensor.share_memory_() 实现共享内存
    - Actor写入obs后，推理服务器直接读取，无需拷贝
    """
    
    def __init__(self, n_actors: int, max_batch_size: int, obs_shape: tuple):
        """
        Args:
            n_actors: Actor数量
            max_batch_size: 每个Actor最大batch大小（通常等于envs_per_actor）
            obs_shape: 单个观察的shape，如 (4, 96, 96) for Atari, (56, 10, 9) for Chinese Chess
        """
        self.n_actors = n_actors
        self.max_batch_size = max_batch_size
        self.obs_shape = obs_shape
        
        # 为每个Actor预分配共享内存缓冲区
        # buffer_shape: [max_batch_size, *obs_shape]
        self.buffers = {}
        for actor_id in range(n_actors):
            buffer = torch.zeros(max_batch_size, *obs_shape, dtype=torch.float32)
            buffer.share_memory_()  # 关键：移动到共享内存
            self.buffers[actor_id] = buffer
        
        logging.info(f"SharedObsBuffer 初始化完成: n_actors={n_actors}, "
                    f"max_batch_size={max_batch_size}, obs_shape={obs_shape}, "
                    f"单个缓冲区大小={max_batch_size * np.prod(obs_shape) * 4 / 1024 / 1024:.2f}MB")
    
    def get_buffer(self, actor_id: int) -> torch.Tensor:
        """获取Actor的缓冲区"""
        return self.buffers[actor_id]


class SharedResponseBuffer:
    """
    共享内存响应缓冲区 - 存储推理结果
    
    存储内容:
    - actions: 动作
    - searched_values: MCTS搜索值
    - predicted_values: 预测值
    - visit_count_distributions: 访问分布
    """
    
    def __init__(self, n_actors: int, max_batch_size: int, action_space_size: int):
        """
        Args:
            n_actors: Actor数量
            max_batch_size: 每个Actor最大batch大小
            action_space_size: 动作空间大小
        """
        self.n_actors = n_actors
        self.max_batch_size = max_batch_size
        self.action_space_size = action_space_size
        
        self.buffers = {}
        for actor_id in range(n_actors):
            # actions: [max_batch_size]
            actions = torch.zeros(max_batch_size, dtype=torch.long)
            actions.share_memory_()
            
            # searched_values: [max_batch_size]
            searched_values = torch.zeros(max_batch_size, dtype=torch.float32)
            searched_values.share_memory_()
            
            # predicted_values: [max_batch_size]
            predicted_values = torch.zeros(max_batch_size, dtype=torch.float32)
            predicted_values.share_memory_()
            
            # visit_count_distributions: [max_batch_size, action_space_size]
            visit_counts = torch.zeros(max_batch_size, action_space_size, dtype=torch.float32)
            visit_counts.share_memory_()
            
            # visit_entropy: [max_batch_size]
            visit_entropy = torch.zeros(max_batch_size, dtype=torch.float32)
            visit_entropy.share_memory_()
            
            self.buffers[actor_id] = {
                'actions': actions,
                'searched_values': searched_values, 
                'predicted_values': predicted_values,
                'visit_counts': visit_counts,
                'visit_entropy': visit_entropy,
            }
        
        logging.info(f"SharedResponseBuffer 初始化完成: n_actors={n_actors}, "
                    f"action_space_size={action_space_size}")
    
    def get_buffer(self, actor_id: int) -> Dict[str, torch.Tensor]:
        """获取Actor的响应缓冲区"""
        return self.buffers[actor_id]


# =============================================================================
# 数据结构定义 - 只传元数据，不传实际数据
# =============================================================================

@dataclass
class InferenceRequest:
    """推理请求 - 只包含元数据，实际数据在共享内存中"""
    actor_id: int                          # Actor ID
    request_id: int                        # 请求ID（用于匹配响应）
    batch_size: int                        # 实际batch大小
    action_mask: List                      # 动作掩码（较小，直接传）
    to_play: List                          # 当前玩家
    temperature: float                     # 温度参数
    epsilon: float                         # epsilon参数
    ready_env_id: List[int]                # ready环境ID列表
    timestep: List = field(default_factory=list)  # 时间步（用于UniZero）


@dataclass 
class InferenceResponse:
    """推理响应 - 只包含元数据，实际数据在共享内存中"""
    actor_id: int                          # Actor ID
    request_id: int                        # 请求ID
    batch_size: int                        # batch大小


@dataclass
class CollectedSegment:
    """收集到的game segment数据"""
    game_segment: GameSegment
    priorities: Optional[np.ndarray]
    done: bool


# =============================================================================
# GPU推理服务器 - 在主进程中运行
# =============================================================================

class GPUInferenceServer:
    """
    GPU推理服务器 - 在主进程的独立线程中运行
    
    功能:
    - 从共享内存读取obs
    - 调用policy.forward()执行推理
    - 将结果写入共享内存
    """
    
    def __init__(
        self,
        policy: namedtuple,
        shared_obs_buffer: SharedObsBuffer,
        shared_response_buffer: SharedResponseBuffer,
        request_queue: mp.Queue,
        response_queues: Dict[int, mp.Queue],
        policy_config: Any,
        logger: Any = None,
    ):
        """
        Args:
            policy: policy.collect_mode
            shared_obs_buffer: 共享内存obs缓冲区
            shared_response_buffer: 共享内存响应缓冲区
            request_queue: 推理请求队列
            response_queues: 每个Actor的响应队列
            policy_config: 策略配置
            logger: 日志器
        """
        self._policy = policy
        self._shared_obs_buffer = shared_obs_buffer
        self._shared_response_buffer = shared_response_buffer
        self._request_queue = request_queue
        self._response_queues = response_queues
        self._policy_config = policy_config
        self._logger = logger
        
        # 从模型参数获取实际设备（支持多卡DDP）
        # policy_config.device 可能是 'cuda' 而不是 'cuda:4'，导致设备不匹配
        try:
            self._device = next(policy._collect_model.parameters()).device
        except (StopIteration, AttributeError):
            self._device = policy_config.device
        
        if self._logger:
            self._logger.info(f"GPUInferenceServer 使用设备: {self._device}")
        
        self._running = False
        self._thread = None
        
        # 统计信息
        self._total_inferences = 0
        self._total_inference_time = 0.0
        
    def start(self):
        """启动推理服务器线程"""
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="GPUInferenceServer")
        self._thread.start()
        if self._logger:
            self._logger.info("GPUInferenceServer 启动")
            
    def stop(self):
        """停止推理服务器"""
        self._running = False
        if self._thread:
            # 放入一个None来唤醒阻塞的get
            try:
                self._request_queue.put(None)
            except:
                pass
            self._thread.join(timeout=5.0)
        if self._logger:
            avg_time = self._total_inference_time / max(1, self._total_inferences)
            self._logger.info(f"GPUInferenceServer 停止. 总推理次数: {self._total_inferences}, "
                            f"平均耗时: {avg_time:.4f}s")
    
    def _run(self):
        """推理服务器主循环"""
        while self._running:
            try:
                # 从队列获取请求，带超时避免死锁
                try:
                    request: Optional[InferenceRequest] = self._request_queue.get(timeout=0.1)
                except:
                    continue
                
                if request is None:
                    continue
                    
                start_time = time.time()
                
                # 从共享内存读取obs并执行推理
                self._do_inference(request)
                
                # 构建响应（只包含元数据）
                response = InferenceResponse(
                    actor_id=request.actor_id,
                    request_id=request.request_id,
                    batch_size=request.batch_size
                )
                
                # 放入对应Actor的响应队列
                if request.actor_id in self._response_queues:
                    self._response_queues[request.actor_id].put(response)
                else:
                    if self._logger:
                        self._logger.error(f"未知的actor_id: {request.actor_id}")
                
                # 更新统计
                self._total_inferences += 1
                self._total_inference_time += time.time() - start_time
                
            except Exception as e:
                if self._logger:
                    self._logger.error(f"GPUInferenceServer 错误: {e}")
                traceback.print_exc()
    
    def _do_inference(self, request: InferenceRequest):
        """
        执行推理 - 从共享内存读取数据，结果写入共享内存
        """
        # 从共享内存读取obs（零拷贝）
        obs_buffer = self._shared_obs_buffer.get_buffer(request.actor_id)
        stack_obs = obs_buffer[:request.batch_size].to(self._device)
        
        # 调用policy.forward()
        policy_output = self._policy.forward(
            stack_obs,
            request.action_mask,
            request.temperature,
            request.to_play,
            request.epsilon,
            ready_env_id=np.array(request.ready_env_id),
            timestep=request.timestep
        )
        
        # 将结果写入共享内存（零拷贝）
        resp_buffer = self._shared_response_buffer.get_buffer(request.actor_id)
        
        for i, env_id in enumerate(request.ready_env_id):
            output = policy_output[env_id]
            resp_buffer['actions'][i] = output['action']
            resp_buffer['searched_values'][i] = output['searched_value']
            resp_buffer['predicted_values'][i] = output['predicted_value']
            
            if 'visit_count_distributions' in output:
                visit_dist = output['visit_count_distributions']
                resp_buffer['visit_counts'][i, :len(visit_dist)] = torch.tensor(visit_dist, dtype=torch.float32)
            
            if 'visit_count_distribution_entropy' in output:
                resp_buffer['visit_entropy'][i] = output['visit_count_distribution_entropy']


# =============================================================================
# Actor进程入口函数
# =============================================================================

def _actor_process_main(
    actor_id: int,
    env_fn: Callable,
    env_configs: List[dict],
    env_manager_cfg: dict,
    policy_config_dict: dict,
    shared_obs_buffer: SharedObsBuffer,
    shared_response_buffer: SharedResponseBuffer,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    segment_queue: mp.Queue,
    stop_event: mp.Event,
    n_episode: int,
    temperature: float,
    epsilon: float,
    seed: int,
):
    """
    Actor子进程入口函数
    
    Args:
        actor_id: Actor ID
        env_fn: 环境创建函数
        env_configs: 该Actor管理的环境配置列表
        env_manager_cfg: env_manager配置
        policy_config_dict: 策略配置（字典形式，可序列化）
        shared_obs_buffer: 共享内存obs缓冲区
        shared_response_buffer: 共享内存响应缓冲区
        request_queue: 请求队列
        response_queue: 响应队列
        segment_queue: segment数据队列
        stop_event: 停止信号
        n_episode: 需要收集的episode数量
        temperature: 温度参数
        epsilon: epsilon参数
        seed: 随机种子
    """
    try:
        # 设置进程名
        import setproctitle
        setproctitle.setproctitle(f"LightZero-Actor-{actor_id}")
    except ImportError:
        pass
    
    # 转换配置为EasyDict
    policy_config = EasyDict(policy_config_dict)
    
    logging.info(f"[Actor-{actor_id}] 进程启动, PID={os.getpid()}")
    
    try:
        # 在子进程中创建环境
        env_manager_cfg_copy = copy.deepcopy(env_manager_cfg)
        if not isinstance(env_manager_cfg_copy, EasyDict):
            env_manager_cfg_copy = EasyDict(env_manager_cfg_copy)
        
        # 默认使用base（同步），因为Actor已经是独立进程
        if 'type' not in env_manager_cfg_copy:
            env_manager_cfg_copy['type'] = 'base'
        
        env_type = env_manager_cfg_copy.get('type', 'base')
        
        # 根据env_manager类型添加必要的默认配置
        if env_type == 'subprocess':
            subprocess_defaults = {
                'episode_num': float('inf'),
                'max_retry': 1,
                'retry_type': 'reset',
                'auto_reset': True,
                'step_timeout': None,
                'reset_timeout': None,
                'retry_waiting_time': 0.1,
                'copy_on_get': True,
                'context': 'spawn',
                'wait_num': float('inf'),
                'step_wait_timeout': None,
                'connect_timeout': 60,
                'reset_inplace': False,
            }
            for key, value in subprocess_defaults.items():
                if key not in env_manager_cfg_copy:
                    env_manager_cfg_copy[key] = value
        else:
            # base env_manager 默认配置
            base_defaults = {
                'episode_num': float('inf'),
                'max_retry': 1,
                'retry_type': 'reset',
                'auto_reset': True,
                'reset_timeout': None,
            }
            for key, value in base_defaults.items():
                if key not in env_manager_cfg_copy:
                    env_manager_cfg_copy[key] = value
        
        env = create_env_manager(
            env_manager_cfg_copy,
            [partial(env_fn, cfg=c) for c in env_configs]
        )
        env.seed(seed + actor_id * 1000)
        env.launch()
        
        env_num = env.env_num
        logging.info(f"[Actor-{actor_id}] 环境创建完成, env_num={env_num}")
        
        # 获取共享内存缓冲区
        obs_buffer = shared_obs_buffer.get_buffer(actor_id)
        resp_buffer = shared_response_buffer.get_buffer(actor_id)
        
        # 运行收集循环
        _actor_collect_loop(
            actor_id=actor_id,
            env=env,
            policy_config=policy_config,
            obs_buffer=obs_buffer,
            resp_buffer=resp_buffer,
            request_queue=request_queue,
            response_queue=response_queue,
            segment_queue=segment_queue,
            stop_event=stop_event,
            n_episode=n_episode,
            temperature=temperature,
            epsilon=epsilon,
        )
        
    except Exception as e:
        logging.error(f"[Actor-{actor_id}] 进程异常: {e}")
        traceback.print_exc()
        # 发送错误信号
        segment_queue.put(('ERROR', actor_id, str(e)))
    finally:
        try:
            env.close()
        except:
            pass
        logging.info(f"[Actor-{actor_id}] 进程退出")


def _actor_collect_loop(
    actor_id: int,
    env: BaseEnvManager,
    policy_config: EasyDict,
    obs_buffer: torch.Tensor,
    resp_buffer: Dict[str, torch.Tensor],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    segment_queue: mp.Queue,
    stop_event: mp.Event,
    n_episode: int,
    temperature: float,
    epsilon: float,
):
    """
    Actor数据收集主循环
    """
    env_num = env.env_num
    request_counter = 0
    collected_episodes = 0
    total_steps = 0
    
    unroll_plus_td_steps = policy_config.num_unroll_steps + policy_config.td_steps
    
    # ============ 初始化 ============
    init_obs = env.ready_obs
    retry_count = 0
    while len(init_obs.keys()) != env_num and retry_count < 100:
        time.sleep(0.01)
        init_obs = env.ready_obs
        retry_count += 1
        
    if len(init_obs.keys()) != env_num:
        raise RuntimeError(f"[Actor-{actor_id}] 无法获取所有环境obs, "
                         f"got {len(init_obs.keys())}, expected {env_num}")
    
    # 初始化数据结构
    action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_num)}
    to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_num)}
    timestep_dict = {i: to_ndarray(init_obs[i].get('timestep', -1)) for i in range(env_num)}
    
    if policy_config.use_ture_chance_label_in_chance_encoder:
        chance_dict = {i: to_ndarray(init_obs[i]['chance']) for i in range(env_num)}
    
    # 创建GameSegment
    game_segments = [
        GameSegment(
            env.action_space,
            game_segment_length=policy_config.game_segment_length,
            config=policy_config
        ) for _ in range(env_num)
    ]
    
    # 观察窗口栈
    observation_window_stack = [
        deque(
            [to_ndarray(init_obs[env_id]['observation']) 
             for _ in range(policy_config.model.frame_stack_num)],
            maxlen=policy_config.model.frame_stack_num
        ) for env_id in range(env_num)
    ]
    
    for env_id in range(env_num):
        game_segments[env_id].reset(observation_window_stack[env_id])
    
    dones = np.array([False for _ in range(env_num)])
    last_game_segments = [None for _ in range(env_num)]
    last_game_priorities = [None for _ in range(env_num)]
    
    # 优先级计算相关
    search_values_lst = [[] for _ in range(env_num)]
    pred_values_lst = [[] for _ in range(env_num)]
    
    # 统计信息
    eps_steps_lst = np.zeros(env_num)
    visit_entropies_lst = np.zeros(env_num)
    
    ready_env_id = set(range(env_num))
    
    logging.info(f"[Actor-{actor_id}] 开始收集循环, 目标episodes={n_episode}")
    
    # ============ 主循环 ============
    while not stop_event.is_set() and collected_episodes < n_episode:
        # 获取ready的obs
        obs = env.ready_obs
        
        if len(obs) == 0:
            time.sleep(0.001)
            continue
            
        current_ready_env_id = list(set(obs.keys()).intersection(ready_env_id))
        if len(current_ready_env_id) == 0:
            time.sleep(0.001)
            continue
        
        batch_size = len(current_ready_env_id)
        
        # 准备推理数据
        stack_obs_list = [game_segments[env_id].get_obs() for env_id in current_ready_env_id]
        action_mask = [action_mask_dict[env_id] for env_id in current_ready_env_id]
        to_play = [to_play_dict[env_id] for env_id in current_ready_env_id]
        timestep = [timestep_dict[env_id] for env_id in current_ready_env_id]
        
        # 预处理观察数据
        stack_obs_array = to_ndarray(stack_obs_list)
        stack_obs_array = prepare_observation(stack_obs_array, policy_config.model.model_type)
        stack_obs_tensor = torch.from_numpy(stack_obs_array).float()
        
        # ============ 写入共享内存 ============
        obs_buffer[:batch_size].copy_(stack_obs_tensor)
        
        # ============ 发送推理请求（只传元数据）============
        request = InferenceRequest(
            actor_id=actor_id,
            request_id=request_counter,
            batch_size=batch_size,
            action_mask=action_mask,
            to_play=to_play,
            temperature=temperature,
            epsilon=epsilon,
            ready_env_id=current_ready_env_id,
            timestep=timestep
        )
        request_counter += 1
        request_queue.put(request)
        
        # ============ 等待推理响应 ============
        response: InferenceResponse = response_queue.get()
        
        if response.request_id != request.request_id:
            raise RuntimeError(f"[Actor-{actor_id}] 响应request_id不匹配: "
                             f"expected {request.request_id}, got {response.request_id}")
        
        # ============ 从共享内存读取结果 ============
        actions_array = resp_buffer['actions'][:batch_size].numpy().copy()
        searched_values_array = resp_buffer['searched_values'][:batch_size].numpy().copy()
        predicted_values_array = resp_buffer['predicted_values'][:batch_size].numpy().copy()
        visit_counts_array = resp_buffer['visit_counts'][:batch_size].numpy().copy()
        visit_entropy_array = resp_buffer['visit_entropy'][:batch_size].numpy().copy()
        
        # 构建policy_output格式
        actions_with_env_id = {}
        value_dict_with_env_id = {}
        pred_value_dict_with_env_id = {}
        distributions_dict_with_env_id = {}
        visit_entropy_dict_with_env_id = {}
        
        for i, env_id in enumerate(current_ready_env_id):
            actions_with_env_id[env_id] = int(actions_array[i])
            value_dict_with_env_id[env_id] = float(searched_values_array[i])
            pred_value_dict_with_env_id[env_id] = float(predicted_values_array[i])
            distributions_dict_with_env_id[env_id] = visit_counts_array[i].tolist()
            visit_entropy_dict_with_env_id[env_id] = float(visit_entropy_array[i])
        
        # ============ 执行环境step ============
        actions = {env_id: actions_with_env_id[env_id] for env_id in current_ready_env_id}
        timesteps = env.step(actions)
        
        # ============ 处理step结果 ============
        for env_id, episode_timestep in timesteps.items():
            if episode_timestep.info.get('abnormal', False):
                env.reset({env_id: None})
                continue
                
            obs_data, reward, done, info = (
                episode_timestep.obs, 
                episode_timestep.reward, 
                episode_timestep.done, 
                episode_timestep.info
            )
            
            # 存储搜索统计
            if not policy_config.collect_with_pure_policy:
                game_segments[env_id].store_search_stats(
                    distributions_dict_with_env_id[env_id],
                    value_dict_with_env_id[env_id]
                )
            
            # 追加transition
            if policy_config.use_ture_chance_label_in_chance_encoder:
                game_segments[env_id].append(
                    actions_with_env_id[env_id], 
                    to_ndarray(obs_data['observation']), 
                    reward,
                    action_mask_dict[env_id],
                    to_play_dict[env_id], 
                    timestep_dict[env_id],
                    chance_dict[env_id]
                )
            else:
                game_segments[env_id].append(
                    actions_with_env_id[env_id],
                    to_ndarray(obs_data['observation']),
                    reward,
                    action_mask_dict[env_id],
                    to_play_dict[env_id],
                    timestep_dict[env_id]
                )
            
            # 更新字典
            action_mask_dict[env_id] = to_ndarray(obs_data['action_mask'])
            to_play_dict[env_id] = to_ndarray(obs_data['to_play'])
            timestep_dict[env_id] = to_ndarray(obs_data.get('timestep', -1))
            if policy_config.use_ture_chance_label_in_chance_encoder:
                chance_dict[env_id] = to_ndarray(obs_data['chance'])
            
            if policy_config.ignore_done:
                dones[env_id] = False
            else:
                dones[env_id] = done
            
            if not policy_config.collect_with_pure_policy:
                visit_entropies_lst[env_id] += visit_entropy_dict_with_env_id[env_id]
            
            eps_steps_lst[env_id] += 1
            total_steps += 1
            
            if policy_config.use_priority:
                pred_values_lst[env_id].append(pred_value_dict_with_env_id[env_id])
                search_values_lst[env_id].append(value_dict_with_env_id[env_id])
            
            # 更新观察窗口
            observation_window_stack[env_id].append(to_ndarray(obs_data['observation']))
            
            # ============ 保存GameSegment ============
            if game_segments[env_id].is_full():
                if last_game_segments[env_id] is not None:
                    _pad_and_save_segment(
                        env_id, last_game_segments, last_game_priorities,
                        game_segments, dones, segment_queue, policy_config, unroll_plus_td_steps
                    )
                
                priorities = _compute_priorities(env_id, pred_values_lst, search_values_lst, policy_config)
                pred_values_lst[env_id] = []
                search_values_lst[env_id] = []
                
                last_game_segments[env_id] = game_segments[env_id]
                last_game_priorities[env_id] = priorities
                
                game_segments[env_id] = GameSegment(
                    env.action_space,
                    game_segment_length=policy_config.game_segment_length,
                    config=policy_config
                )
                game_segments[env_id].reset(observation_window_stack[env_id])
            
            # ============ Episode结束处理 ============
            if episode_timestep.done:
                collected_episodes += 1
                
                # 保存最后的segment
                if last_game_segments[env_id] is not None:
                    _pad_and_save_segment(
                        env_id, last_game_segments, last_game_priorities,
                        game_segments, dones, segment_queue, policy_config, unroll_plus_td_steps
                    )
                
                priorities = _compute_priorities(env_id, pred_values_lst, search_values_lst, policy_config)
                game_segments[env_id].game_segment_to_array()
                
                if len(game_segments[env_id].reward_segment) != 0:
                    segment_queue.put(CollectedSegment(
                        game_segment=game_segments[env_id],
                        priorities=priorities,
                        done=dones[env_id]
                    ))
                
                # 重置该环境
                pred_values_lst[env_id] = []
                search_values_lst[env_id] = []
                eps_steps_lst[env_id] = 0
                visit_entropies_lst[env_id] = 0
                
                # 如果还需要继续收集，重新初始化
                if collected_episodes < n_episode:
                    # 等待env reset完成
                    reset_obs = None
                    for _ in range(100):
                        reset_obs = env.ready_obs
                        if env_id in reset_obs:
                            break
                        time.sleep(0.01)
                    
                    if reset_obs and env_id in reset_obs:
                        action_mask_dict[env_id] = to_ndarray(reset_obs[env_id]['action_mask'])
                        to_play_dict[env_id] = to_ndarray(reset_obs[env_id]['to_play'])
                        timestep_dict[env_id] = to_ndarray(reset_obs[env_id].get('timestep', -1))
                        if policy_config.use_ture_chance_label_in_chance_encoder:
                            chance_dict[env_id] = to_ndarray(reset_obs[env_id]['chance'])
                        
                        game_segments[env_id] = GameSegment(
                            env.action_space,
                            game_segment_length=policy_config.game_segment_length,
                            config=policy_config
                        )
                        observation_window_stack[env_id] = deque(
                            [reset_obs[env_id]['observation'] 
                             for _ in range(policy_config.model.frame_stack_num)],
                            maxlen=policy_config.model.frame_stack_num
                        )
                        game_segments[env_id].reset(observation_window_stack[env_id])
                        last_game_segments[env_id] = None
                        last_game_priorities[env_id] = None
    
    logging.info(f"[Actor-{actor_id}] 收集完成: {collected_episodes} episodes, {total_steps} steps")


def _compute_priorities(env_id: int, pred_values_lst: List, search_values_lst: List, 
                       policy_config: EasyDict) -> Optional[np.ndarray]:
    """计算优先级"""
    if policy_config.use_priority and len(pred_values_lst[env_id]) > 0:
        pred_values = torch.tensor(pred_values_lst[env_id], dtype=torch.float32)
        search_values = torch.tensor(search_values_lst[env_id], dtype=torch.float32)
        priorities = L1Loss(reduction='none')(pred_values, search_values).numpy() + 1e-6
    else:
        priorities = None
    return priorities


def _pad_and_save_segment(
    env_id: int,
    last_game_segments: List,
    last_game_priorities: List,
    game_segments: List,
    dones: np.ndarray,
    segment_queue: mp.Queue,
    policy_config: EasyDict,
    unroll_plus_td_steps: int,
):
    """填充并保存segment"""
    beg_index = policy_config.model.frame_stack_num
    end_index = beg_index + policy_config.num_unroll_steps + policy_config.td_steps
    
    pad_obs_lst = game_segments[env_id].obs_segment[beg_index:end_index]
    
    beg_index = 0
    end_index = beg_index + policy_config.num_unroll_steps + policy_config.td_steps
    pad_action_lst = game_segments[env_id].action_segment[beg_index:end_index]
    pad_child_visits_lst = game_segments[env_id].child_visit_segment[
        :policy_config.num_unroll_steps + policy_config.td_steps
    ]
    
    beg_index = 0
    end_index = beg_index + unroll_plus_td_steps - 1
    pad_reward_lst = game_segments[env_id].reward_segment[beg_index:end_index]
    
    beg_index = 0
    end_index = beg_index + unroll_plus_td_steps
    pad_root_values_lst = game_segments[env_id].root_value_segment[beg_index:end_index]
    
    if policy_config.gumbel_algo:
        pad_improved_policy_prob = game_segments[env_id].improved_policy_probs[beg_index:end_index]
        last_game_segments[env_id].pad_over(
            pad_obs_lst, pad_reward_lst, pad_action_lst, 
            pad_root_values_lst, pad_child_visits_lst,
            next_segment_improved_policy=pad_improved_policy_prob
        )
    else:
        last_game_segments[env_id].pad_over(
            pad_obs_lst, pad_reward_lst, pad_action_lst,
            pad_root_values_lst, pad_child_visits_lst
        )
    
    last_game_segments[env_id].game_segment_to_array()
    
    segment_queue.put(CollectedSegment(
        game_segment=last_game_segments[env_id],
        priorities=last_game_priorities[env_id],
        done=dones[env_id]
    ))
    
    last_game_segments[env_id] = None
    last_game_priorities[env_id] = None


# =============================================================================
# 多Actor收集器 - 主协调器
# =============================================================================

@SERIAL_COLLECTOR_REGISTRY.register('multi_actor_muzero')
class MultiActorMuZeroCollector(ISerialCollector):
    """
    多Actor并行收集器 - 多进程 + 共享内存实现
    
    核心优势:
    - 真正的多进程并行，绕过GIL限制
    - 共享内存零拷贝，无序列化开销
    - N个Actor并行运行，GPU持续推理
    
    使用方式:
    - 接口与MuZeroCollector完全一致
    - 通过policy_config.n_actors配置Actor数量
    - 通过policy_config.envs_per_actor配置每个Actor管理的环境数量
    """
    
    config = dict()
    
    def __init__(
        self,
        collect_print_freq: int = 100,
        env: BaseEnvManager = None,
        policy: namedtuple = None,
        tb_logger: 'SummaryWriter' = None,
        exp_name: Optional[str] = 'default_experiment',
        instance_name: Optional[str] = 'multi_actor_collector',
        policy_config: 'policy_config' = None,
        env_fn: Callable = None,
        env_config: List[dict] = None,
        env_manager_cfg: dict = None,
    ) -> None:
        """
        Args:
            collect_print_freq: 打印频率
            env: 原始env_manager（用于获取action_space等信息）
            policy: policy.collect_mode
            tb_logger: TensorBoard logger
            exp_name: 实验名称
            instance_name: 实例名称
            policy_config: 策略配置
            env_fn: 环境创建函数
            env_config: 环境配置列表
            env_manager_cfg: env_manager配置
        """
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = collect_print_freq
        self._timer = EasyTimer()
        self._end_flag = False
        
        self._rank = get_rank()
        self._world_size = get_world_size()
        
        # 初始化logger
        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                    name=self._instance_name,
                    need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                    name=self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                name=self._instance_name,
                need_tb=False
            )
            self._tb_logger = None
        
        self.policy_config = policy_config
        self._policy = policy
        
        # 多Actor配置
        self._n_actors = getattr(policy_config, 'n_actors', 4)
        self._envs_per_actor = getattr(policy_config, 'envs_per_actor', 8)
        
        # 保存环境创建信息
        self._env_fn = env_fn
        self._env_config = env_config
        self._original_env = env
        self._env_manager_cfg = env_manager_cfg if env_manager_cfg else {}
        
        # 获取obs_shape和action_space_size
        self._obs_shape = self._get_obs_shape()
        self._action_space_size = self._get_action_space_size()
        
        # 初始化共享内存缓冲区
        self._shared_obs_buffer = SharedObsBuffer(
            n_actors=self._n_actors,
            max_batch_size=self._envs_per_actor,
            obs_shape=self._obs_shape
        )
        self._shared_response_buffer = SharedResponseBuffer(
            n_actors=self._n_actors,
            max_batch_size=self._envs_per_actor,
            action_space_size=self._action_space_size
        )
        
        # 进程间通信队列
        self._request_queue: mp.Queue = mp.Queue()
        self._response_queues: Dict[int, mp.Queue] = {i: mp.Queue() for i in range(self._n_actors)}
        self._segment_queue: mp.Queue = mp.Queue()
        
        # 停止信号
        self._stop_event: mp.Event = mp.Event()
        
        # 进程列表
        self._actor_processes: List[mp.Process] = []
        
        # 组件
        self._inference_server: Optional[GPUInferenceServer] = None
        
        # 统计
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._episode_info = []
        
        # 数据池
        self.game_segment_pool = deque(maxlen=int(1e6))
        self.unroll_plus_td_steps = policy_config.num_unroll_steps + policy_config.td_steps
        
        self._logger.info(f"MultiActorMuZeroCollector 初始化完成: "
                         f"n_actors={self._n_actors}, envs_per_actor={self._envs_per_actor}, "
                         f"obs_shape={self._obs_shape}, action_space_size={self._action_space_size}")
    
    def _get_obs_shape(self) -> tuple:
        """获取观察shape"""
        # 从policy_config获取
        model_cfg = self.policy_config.model
        obs_shape = model_cfg.observation_shape
        
        # 对于需要frame_stack的情况，obs_shape已经包含了通道数
        # 例如 (4, 96, 96) for Atari with frame_stack=4
        # 或者 (56, 10, 9) for Chinese Chess
        return tuple(obs_shape)
    
    def _get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return self.policy_config.model.action_space_size
    
    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """重置收集器"""
        if _env is not None:
            self._original_env = _env
        if _policy is not None:
            self._policy = _policy
        
        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False
        self.game_segment_pool.clear()
    
    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """重置环境"""
        if _env is not None:
            self._original_env = _env
    
    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """重置策略"""
        if _policy is not None:
            self._policy = _policy
        self._policy.reset()
    
    @property
    def envstep(self) -> int:
        """返回总环境步数"""
        return self._total_envstep_count
    
    def close(self) -> None:
        """关闭收集器"""
        if self._end_flag:
            return
        self._end_flag = True
        
        # 发送停止信号
        self._stop_event.set()
        
        # 停止推理服务器
        if self._inference_server:
            self._inference_server.stop()
        
        # 停止所有Actor进程
        for process in self._actor_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        
        # 关闭原始环境
        if self._original_env:
            try:
                self._original_env.close()
            except:
                pass
        
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()
        
        self._logger.info("MultiActorMuZeroCollector 关闭完成")
    
    def __del__(self):
        self.close()
    
    def collect(
        self,
        n_episode: Optional[int] = None,
        train_iter: int = 0,
        policy_kwargs: Optional[dict] = None,
        collect_with_pure_policy: bool = False
    ) -> List[Any]:
        """
        收集数据 - 接口与MuZeroCollector完全一致
        
        Args:
            n_episode: 要收集的episode数量
            train_iter: 当前训练迭代
            policy_kwargs: 策略参数（temperature, epsilon）
            collect_with_pure_policy: 是否使用纯策略收集
            
        Returns:
            return_data: [game_segments, meta_data]
        """
        if n_episode is None:
            n_episode = self._policy.get_attribute('cfg').get('n_episode', self._n_actors * self._envs_per_actor)
        
        if policy_kwargs is None:
            policy_kwargs = {'temperature': 1.0, 'epsilon': 0.0}
        
        temperature = policy_kwargs.get('temperature', 1.0)
        epsilon = policy_kwargs.get('epsilon', 0.0)
        
        self._logger.info(f"开始多Actor数据采集: n_episode={n_episode}, "
                         f"n_actors={self._n_actors}, temperature={temperature:.4f}, epsilon={epsilon:.4f}")
        
        start_time = time.time()
        
        # 清空停止信号
        self._stop_event.clear()
        
        # 清空队列
        while not self._segment_queue.empty():
            try:
                self._segment_queue.get_nowait()
            except:
                break
        
        # 启动GPU推理服务器
        self._inference_server = GPUInferenceServer(
            policy=self._policy,
            shared_obs_buffer=self._shared_obs_buffer,
            shared_response_buffer=self._shared_response_buffer,
            request_queue=self._request_queue,
            response_queues=self._response_queues,
            policy_config=self.policy_config,
            logger=self._logger
        )
        self._inference_server.start()
        
        # 计算每个Actor需要收集的episode数量
        episodes_per_actor = n_episode // self._n_actors
        remainder = n_episode % self._n_actors
        
        # 启动Actor进程
        self._actor_processes = []
        for actor_id in range(self._n_actors):
            actor_episodes = episodes_per_actor + (1 if actor_id < remainder else 0)
            
            # 获取该Actor的环境配置
            start_idx = actor_id * self._envs_per_actor
            end_idx = start_idx + self._envs_per_actor
            actor_env_configs = list(self._env_config[start_idx:end_idx])
            
            # 如果配置不够，循环使用
            while len(actor_env_configs) < self._envs_per_actor:
                remaining = self._envs_per_actor - len(actor_env_configs)
                actor_env_configs.extend(list(self._env_config[:remaining]))
            
            # 创建进程
            process = mp.Process(
                target=_actor_process_main,
                args=(
                    actor_id,
                    self._env_fn,
                    actor_env_configs,
                    dict(self._env_manager_cfg),
                    dict(self.policy_config),
                    self._shared_obs_buffer,
                    self._shared_response_buffer,
                    self._request_queue,
                    self._response_queues[actor_id],
                    self._segment_queue,
                    self._stop_event,
                    actor_episodes,
                    temperature,
                    epsilon,
                    self.policy_config.seed if hasattr(self.policy_config, 'seed') else 0,
                ),
                daemon=True,
                name=f"Actor-{actor_id}"
            )
            process.start()
            self._actor_processes.append(process)
        
        self._logger.info(f"已启动 {len(self._actor_processes)} 个Actor进程")
        
        # 等待所有Actor完成并收集数据
        collected_segments = []
        
        while True:
            # 检查是否所有进程都已完成
            all_done = all(not p.is_alive() for p in self._actor_processes)
            
            # 从segment队列收集数据
            while not self._segment_queue.empty():
                try:
                    item = self._segment_queue.get_nowait()
                    
                    # 检查是否是错误信号
                    if isinstance(item, tuple) and len(item) == 3 and item[0] == 'ERROR':
                        _, actor_id, error_msg = item
                        self._logger.error(f"Actor-{actor_id} 发生错误: {error_msg}")
                        continue
                    
                    if isinstance(item, CollectedSegment):
                        collected_segments.append(item)
                except:
                    break
            
            if all_done:
                # 收集剩余数据
                time.sleep(0.1)
                while not self._segment_queue.empty():
                    try:
                        item = self._segment_queue.get_nowait()
                        if isinstance(item, CollectedSegment):
                            collected_segments.append(item)
                    except:
                        break
                break
            
            time.sleep(0.01)
        
        # 停止推理服务器
        self._inference_server.stop()
        
        # 清理进程
        for process in self._actor_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        self._actor_processes.clear()
        
        # 整理返回数据
        for segment in collected_segments:
            self.game_segment_pool.append((segment.game_segment, segment.priorities, segment.done))
        
        return_data = [
            [self.game_segment_pool[i][0] for i in range(len(self.game_segment_pool))],
            [{
                'priorities': self.game_segment_pool[i][1],
                'done': self.game_segment_pool[i][2],
                'unroll_plus_td_steps': self.unroll_plus_td_steps
            } for i in range(len(self.game_segment_pool))]
        ]
        
        # 更新统计
        collected_duration = time.time() - start_time
        self._total_episode_count += len(collected_segments)
        self._total_duration += collected_duration
        
        # 统计envstep
        for segment in collected_segments:
            self._total_envstep_count += len(segment.game_segment.reward_segment)
        
        self._logger.info(f"多Actor采集完成: {len(collected_segments)} segments, "
                         f"{collected_duration:.2f}s, total_envstep={self._total_envstep_count}")
        
        # 清空pool
        self.game_segment_pool.clear()
        
        # 输出日志
        self._output_log(train_iter)
        
        return return_data
    
    def _output_log(self, train_iter: int) -> None:
        """输出日志"""
        if self._rank != 0:
            return
        
        if self._tb_logger and self._total_episode_count > 0:
            info = {
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
            }
            for k, v in info.items():
                self._tb_logger.add_scalar(f'{self._instance_name}_iter/{k}', v, train_iter)
