"""
多Actor并行采集器 - 通用设计，支持所有MuZero系列算法

架构设计:
┌─────────────────────────────────────────────────────────────────────┐
│  MultiActorMuZeroCollector (主协调器)                                │
│      ├── GPUInferenceServer (单独线程，处理GPU推理)                   │
│      │       └── policy.forward()                                   │
│      │                                                              │
│      └── ActorWorker × N (多线程)                                    │
│              └── EnvManager (每个Actor管理自己的envs)                 │
│                                                                     │
│  工作流程:                                                           │
│  1. 每个ActorWorker独立运行自己的envs                                 │
│  2. 当某个Actor的所有envs ready时，将obs放入推理队列                   │
│  3. GPUInferenceServer从队列取出请求，执行推理，返回结果               │
│  4. Actor收到结果后执行actions，继续循环                              │
│  5. 数据汇总到共享的game_segment_pool                                │
└─────────────────────────────────────────────────────────────────────┘

优势:
- GPU不再空闲等待，总有Actor ready可以推理
- 绕过单核CPU性能瓶颈
- 通用设计，支持所有MuZero系列算法
"""

import os
import time
import copy
import queue
import threading
from collections import deque, namedtuple
from typing import Optional, Any, List, Dict, Callable, Tuple, Union
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import wandb
from easydict import EasyDict
from ding.envs import BaseEnvManager, create_env_manager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size, allreduce_data
from ding.worker.collector.base_serial_collector import ISerialCollector
from torch.nn import L1Loss

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class InferenceRequest:
    """推理请求数据结构"""
    actor_id: int                          # Actor ID
    request_id: int                        # 请求ID（用于匹配响应）
    stack_obs: torch.Tensor                # 批量观察 [B, C, H, W]
    action_mask: List                      # 动作掩码
    to_play: List                          # 当前玩家
    temperature: float                     # 温度参数
    epsilon: float                         # epsilon参数
    ready_env_id: np.ndarray               # ready环境ID
    timestep: List = field(default_factory=list)  # 时间步（用于UniZero）


@dataclass 
class InferenceResponse:
    """推理响应数据结构"""
    actor_id: int                          # Actor ID
    request_id: int                        # 请求ID
    policy_output: Dict                    # policy.forward()的输出


@dataclass
class CollectedSegment:
    """收集到的game segment数据"""
    game_segment: GameSegment
    priorities: Optional[np.ndarray]
    done: bool


# =============================================================================
# GPU推理服务器
# =============================================================================

class GPUInferenceServer:
    """
    GPU推理服务器 - 独立线程运行
    
    功能:
    - 从请求队列中获取推理请求
    - 调用policy.forward()执行推理
    - 将结果放入对应的响应队列
    
    设计要点:
    - 线程安全
    - 支持任意policy（MuZero, EfficientZero, Gumbel等）
    """
    
    def __init__(
        self,
        policy: namedtuple,
        request_queue: queue.Queue,
        response_queues: Dict[int, queue.Queue],  # actor_id -> response_queue
        policy_config: Any,
        logger: Any = None,
    ):
        """
        Args:
            policy: policy.collect_mode，支持forward方法
            request_queue: 推理请求队列
            response_queues: 每个Actor的响应队列 {actor_id: queue}
            policy_config: 策略配置
            logger: 日志器
        """
        self._policy = policy
        self._request_queue = request_queue
        self._response_queues = response_queues
        self._policy_config = policy_config
        self._logger = logger
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._total_inferences = 0
        self._total_inference_time = 0.0
        
    def start(self):
        """启动推理服务器线程"""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="GPUInferenceServer")
        self._thread.start()
        if self._logger:
            self._logger.info("GPUInferenceServer started")
            
    def stop(self):
        """停止推理服务器"""
        self._running = False
        if self._thread:
            # 放入一个None来唤醒阻塞的get
            self._request_queue.put(None)
            self._thread.join(timeout=5.0)
        if self._logger:
            self._logger.info(f"GPUInferenceServer stopped. Total inferences: {self._total_inferences}, "
                            f"Avg time: {self._total_inference_time / max(1, self._total_inferences):.4f}s")
    
    def _run(self):
        """推理服务器主循环"""
        while self._running:
            try:
                # 从队列获取请求，带超时避免死锁
                request: Optional[InferenceRequest] = self._request_queue.get(timeout=0.1)
                
                if request is None:
                    continue
                    
                start_time = time.time()
                
                # 执行推理
                policy_output = self._do_inference(request)
                
                # 构建响应
                response = InferenceResponse(
                    actor_id=request.actor_id,
                    request_id=request.request_id,
                    policy_output=policy_output
                )
                
                # 放入对应Actor的响应队列
                if request.actor_id in self._response_queues:
                    self._response_queues[request.actor_id].put(response)
                else:
                    if self._logger:
                        self._logger.error(f"Unknown actor_id: {request.actor_id}")
                
                # 更新统计
                self._total_inferences += 1
                self._total_inference_time += time.time() - start_time
                
            except queue.Empty:
                continue
            except Exception as e:
                if self._logger:
                    self._logger.error(f"GPUInferenceServer error: {e}")
                import traceback
                traceback.print_exc()
    
    def _do_inference(self, request: InferenceRequest) -> Dict:
        """
        执行推理 - 调用policy.forward()
        
        这个方法是通用的，支持所有MuZero系列算法
        """
        policy_output = self._policy.forward(
            request.stack_obs,
            request.action_mask,
            request.temperature,
            request.to_play,
            request.epsilon,
            ready_env_id=request.ready_env_id,
            timestep=request.timestep
        )
        return policy_output


# =============================================================================
# Actor Worker - 单个Actor的工作逻辑
# =============================================================================

class ActorWorker:
    """
    单个Actor Worker - 管理自己的一组envs
    
    功能:
    - 管理自己的env_manager和game_segments
    - 当envs全部ready时，发送推理请求
    - 接收推理结果，执行actions
    - 收集game_segment数据
    
    设计要点:
    - 与MuZeroCollector的核心逻辑保持一致
    - 通过队列与GPUInferenceServer通信
    """
    
    def __init__(
        self,
        actor_id: int,
        env: BaseEnvManager,
        request_queue: queue.Queue,
        response_queue: queue.Queue,
        segment_queue: queue.Queue,  # 收集到的segment放入此队列
        policy_config: Any,
        policy_reset_fn: Callable,   # policy.reset的函数
        logger: Any = None,
    ):
        """
        Args:
            actor_id: Actor的唯一ID
            env: 该Actor管理的env_manager
            request_queue: 发送推理请求的队列
            response_queue: 接收推理响应的队列
            segment_queue: 收集到的segment放入此队列
            policy_config: 策略配置
            policy_reset_fn: policy.reset的函数（用于重置policy状态）
            logger: 日志器
        """
        self._actor_id = actor_id
        self._env = env
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._segment_queue = segment_queue
        self._policy_config = policy_config
        self._policy_reset_fn = policy_reset_fn
        self._logger = logger
        
        # 环境信息
        self._env_num = self._env.env_num
        
        # 请求计数器
        self._request_counter = 0
        
        # 运行状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._total_episodes = 0
        self._total_steps = 0
        
        # 配置参数
        self.unroll_plus_td_steps = policy_config.num_unroll_steps + policy_config.td_steps
        
    def start(self, n_episode: int, temperature: float, epsilon: float):
        """
        启动Actor Worker线程
        
        Args:
            n_episode: 需要收集的episode数量
            temperature: 温度参数
            epsilon: epsilon参数
        """
        self._running = True
        self._target_episodes = n_episode
        self._temperature = temperature
        self._epsilon = epsilon
        self._collected_episodes = 0
        
        self._thread = threading.Thread(
            target=self._run, 
            daemon=True, 
            name=f"ActorWorker-{self._actor_id}"
        )
        self._thread.start()
        
    def stop(self):
        """停止Actor Worker"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
            
    def is_alive(self) -> bool:
        """检查线程是否存活"""
        return self._thread is not None and self._thread.is_alive()
    
    def is_done(self) -> bool:
        """检查是否完成收集任务"""
        return self._collected_episodes >= self._target_episodes
    
    def _run(self):
        """Actor Worker主循环"""
        try:
            self._collect_loop()
        except Exception as e:
            if self._logger:
                self._logger.error(f"ActorWorker-{self._actor_id} error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            
    def _collect_loop(self):
        """
        数据收集主循环 - 核心逻辑与MuZeroCollector.collect()保持一致
        """
        env_nums = self._env_num
        
        # ============ 初始化 ============
        init_obs = self._env.ready_obs
        retry_count = 0
        while len(init_obs.keys()) != self._env_num and retry_count < 100:
            time.sleep(0.01)
            init_obs = self._env.ready_obs
            retry_count += 1
            
        if len(init_obs.keys()) != self._env_num:
            raise RuntimeError(f"ActorWorker-{self._actor_id}: Failed to get all env obs, "
                             f"got {len(init_obs.keys())}, expected {self._env_num}")
        
        # 初始化数据结构
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
        to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
        timestep_dict = {i: to_ndarray(init_obs[i].get('timestep', -1)) for i in range(env_nums)}
        
        if self._policy_config.use_ture_chance_label_in_chance_encoder:
            chance_dict = {i: to_ndarray(init_obs[i]['chance']) for i in range(env_nums)}
        
        # 创建GameSegment
        game_segments = [
            GameSegment(
                self._env.action_space,
                game_segment_length=self._policy_config.game_segment_length,
                config=self._policy_config
            ) for _ in range(env_nums)
        ]
        
        # 观察窗口栈
        observation_window_stack = [
            deque(
                [to_ndarray(init_obs[env_id]['observation']) 
                 for _ in range(self._policy_config.model.frame_stack_num)],
                maxlen=self._policy_config.model.frame_stack_num
            ) for env_id in range(env_nums)
        ]
        
        for env_id in range(env_nums):
            game_segments[env_id].reset(observation_window_stack[env_id])
        
        dones = np.array([False for _ in range(env_nums)])
        last_game_segments = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]
        
        # 优先级计算相关
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]
        if self._policy_config.gumbel_algo:
            improved_policy_lst = [[] for _ in range(env_nums)]
        
        # 统计信息
        eps_steps_lst = np.zeros(env_nums)
        visit_entropies_lst = np.zeros(env_nums)
        
        ready_env_id = set(range(env_nums))
        
        # ============ 主循环 ============
        while self._running and self._collected_episodes < self._target_episodes:
            # 获取ready的obs
            obs = self._env.ready_obs
            
            if len(obs) == 0:
                time.sleep(0.001)
                continue
                
            current_ready_env_id = set(obs.keys()).intersection(ready_env_id)
            if len(current_ready_env_id) == 0:
                time.sleep(0.001)
                continue
            
            # 准备推理数据
            stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in current_ready_env_id}
            stack_obs_list = list(stack_obs.values())
            
            action_mask = [action_mask_dict[env_id] for env_id in current_ready_env_id]
            to_play = [to_play_dict[env_id] for env_id in current_ready_env_id]
            timestep = [timestep_dict[env_id] for env_id in current_ready_env_id]
            
            # 预处理观察数据
            stack_obs_array = to_ndarray(stack_obs_list)
            stack_obs_array = prepare_observation(stack_obs_array, self._policy_config.model.model_type)
            stack_obs_tensor = torch.from_numpy(stack_obs_array).to(self._policy_config.device)
            
            # ============ 发送推理请求 ============
            request = InferenceRequest(
                actor_id=self._actor_id,
                request_id=self._request_counter,
                stack_obs=stack_obs_tensor,
                action_mask=action_mask,
                to_play=to_play,
                temperature=self._temperature,
                epsilon=self._epsilon,
                ready_env_id=np.array(list(current_ready_env_id)),
                timestep=timestep
            )
            self._request_counter += 1
            self._request_queue.put(request)
            
            # ============ 等待推理响应 ============
            response: InferenceResponse = self._response_queue.get()
            
            if response.request_id != request.request_id:
                raise RuntimeError(f"Response request_id mismatch: expected {request.request_id}, "
                                 f"got {response.request_id}")
            
            policy_output = response.policy_output
            
            # 提取策略输出
            actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
            value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
            pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}
            
            if not self._policy_config.collect_with_pure_policy:
                distributions_dict_with_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}
                visit_entropy_dict_with_env_id = {k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()}
                
                if self._policy_config.gumbel_algo:
                    improved_policy_dict_with_env_id = {k: v['improved_policy_probs'] for k, v in policy_output.items()}
            
            if self._policy_config.sampled_algo:
                root_sampled_actions_dict_with_env_id = {k: v['root_sampled_actions'] for k, v in policy_output.items()}
            
            # ============ 执行环境step ============
            actions = {env_id: actions_with_env_id[env_id] for env_id in current_ready_env_id}
            timesteps = self._env.step(actions)
            
            # ============ 处理step结果 ============
            for env_id, episode_timestep in timesteps.items():
                if episode_timestep.info.get('abnormal', False):
                    self._env.reset({env_id: None})
                    self._policy_reset_fn([env_id])
                    continue
                    
                obs_data, reward, done, info = (
                    episode_timestep.obs, 
                    episode_timestep.reward, 
                    episode_timestep.done, 
                    episode_timestep.info
                )
                
                # 存储搜索统计
                if not self._policy_config.collect_with_pure_policy:
                    if self._policy_config.sampled_algo:
                        game_segments[env_id].store_search_stats(
                            distributions_dict_with_env_id[env_id], 
                            value_dict_with_env_id[env_id],
                            root_sampled_actions_dict_with_env_id[env_id]
                        )
                    elif self._policy_config.gumbel_algo:
                        game_segments[env_id].store_search_stats(
                            distributions_dict_with_env_id[env_id],
                            value_dict_with_env_id[env_id],
                            improved_policy=improved_policy_dict_with_env_id[env_id]
                        )
                    else:
                        game_segments[env_id].store_search_stats(
                            distributions_dict_with_env_id[env_id],
                            value_dict_with_env_id[env_id]
                        )
                
                # 追加transition
                if self._policy_config.use_ture_chance_label_in_chance_encoder:
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
                if self._policy_config.use_ture_chance_label_in_chance_encoder:
                    chance_dict[env_id] = to_ndarray(obs_data['chance'])
                
                if self._policy_config.ignore_done:
                    dones[env_id] = False
                else:
                    dones[env_id] = done
                
                if not self._policy_config.collect_with_pure_policy:
                    visit_entropies_lst[env_id] += visit_entropy_dict_with_env_id[env_id]
                
                eps_steps_lst[env_id] += 1
                self._total_steps += 1
                
                if self._policy_config.use_priority:
                    pred_values_lst[env_id].append(pred_value_dict_with_env_id[env_id])
                    search_values_lst[env_id].append(value_dict_with_env_id[env_id])
                
                # 更新观察窗口
                observation_window_stack[env_id].append(to_ndarray(obs_data['observation']))
                
                # ============ 保存GameSegment ============
                if game_segments[env_id].is_full():
                    if last_game_segments[env_id] is not None:
                        self._pad_and_save_segment(
                            env_id, last_game_segments, last_game_priorities,
                            game_segments, dones
                        )
                    
                    priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                    pred_values_lst[env_id] = []
                    search_values_lst[env_id] = []
                    
                    last_game_segments[env_id] = game_segments[env_id]
                    last_game_priorities[env_id] = priorities
                    
                    game_segments[env_id] = GameSegment(
                        self._env.action_space,
                        game_segment_length=self._policy_config.game_segment_length,
                        config=self._policy_config
                    )
                    game_segments[env_id].reset(observation_window_stack[env_id])
                
                # ============ Episode结束处理 ============
                if episode_timestep.done:
                    self._collected_episodes += 1
                    self._total_episodes += 1
                    
                    # 保存最后的segment
                    if last_game_segments[env_id] is not None:
                        self._pad_and_save_segment(
                            env_id, last_game_segments, last_game_priorities,
                            game_segments, dones
                        )
                    
                    priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                    game_segments[env_id].game_segment_to_array()
                    
                    if len(game_segments[env_id].reward_segment) != 0:
                        self._segment_queue.put(CollectedSegment(
                            game_segment=game_segments[env_id],
                            priorities=priorities,
                            done=dones[env_id]
                        ))
                    
                    # 重置该环境
                    pred_values_lst[env_id] = []
                    search_values_lst[env_id] = []
                    eps_steps_lst[env_id] = 0
                    visit_entropies_lst[env_id] = 0
                    
                    self._policy_reset_fn([env_id])
                    
                    # 如果还需要继续收集，重新初始化
                    if self._collected_episodes < self._target_episodes:
                        # 等待env reset完成
                        reset_obs = None
                        for _ in range(100):
                            reset_obs = self._env.ready_obs
                            if env_id in reset_obs:
                                break
                            time.sleep(0.01)
                        
                        if reset_obs and env_id in reset_obs:
                            action_mask_dict[env_id] = to_ndarray(reset_obs[env_id]['action_mask'])
                            to_play_dict[env_id] = to_ndarray(reset_obs[env_id]['to_play'])
                            timestep_dict[env_id] = to_ndarray(reset_obs[env_id].get('timestep', -1))
                            if self._policy_config.use_ture_chance_label_in_chance_encoder:
                                chance_dict[env_id] = to_ndarray(reset_obs[env_id]['chance'])
                            
                            game_segments[env_id] = GameSegment(
                                self._env.action_space,
                                game_segment_length=self._policy_config.game_segment_length,
                                config=self._policy_config
                            )
                            observation_window_stack[env_id] = deque(
                                [reset_obs[env_id]['observation'] 
                                 for _ in range(self._policy_config.model.frame_stack_num)],
                                maxlen=self._policy_config.model.frame_stack_num
                            )
                            game_segments[env_id].reset(observation_window_stack[env_id])
                            last_game_segments[env_id] = None
                            last_game_priorities[env_id] = None
    
    def _compute_priorities(self, env_id: int, pred_values_lst: List, search_values_lst: List) -> Optional[np.ndarray]:
        """计算优先级"""
        if self._policy_config.use_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[env_id])).to(
                self._policy_config.device).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[env_id])).to(
                self._policy_config.device).float().view(-1)
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + 1e-6
        else:
            priorities = None
        return priorities
    
    def _pad_and_save_segment(
        self, 
        env_id: int,
        last_game_segments: List,
        last_game_priorities: List,
        game_segments: List,
        dones: np.ndarray
    ):
        """填充并保存segment"""
        beg_index = self._policy_config.model.frame_stack_num
        end_index = beg_index + self._policy_config.num_unroll_steps + self._policy_config.td_steps
        
        pad_obs_lst = game_segments[env_id].obs_segment[beg_index:end_index]
        
        beg_index = 0
        end_index = beg_index + self._policy_config.num_unroll_steps + self._policy_config.td_steps
        pad_action_lst = game_segments[env_id].action_segment[beg_index:end_index]
        pad_child_visits_lst = game_segments[env_id].child_visit_segment[
            :self._policy_config.num_unroll_steps + self._policy_config.td_steps
        ]
        
        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps - 1
        pad_reward_lst = game_segments[env_id].reward_segment[beg_index:end_index]
        
        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps
        pad_root_values_lst = game_segments[env_id].root_value_segment[beg_index:end_index]
        
        if self._policy_config.gumbel_algo:
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
        
        self._segment_queue.put(CollectedSegment(
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
    多Actor并行收集器 - 替代MuZeroCollector
    
    核心优势:
    - N个Actor并行运行，谁先ready谁先推理
    - GPU不再空闲等待，连续推理
    - 绕过单核CPU性能瓶颈
    
    使用方式:
    - 接口与MuZeroCollector完全一致
    - 通过policy_config.n_actors配置Actor数量
    - 通过policy_config.envs_per_actor配置每个Actor管理的环境数量
    
    支持的算法:
    - MuZero, EfficientZero, Gumbel MuZero, Sampled MuZero等所有MuZero系列
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
        env_fn: Callable = None,              # 环境创建函数
        env_config: List[dict] = None,        # 环境配置列表
    ) -> None:
        """
        Args:
            collect_print_freq: 打印频率
            env: 原始env_manager（将被忽略，使用env_fn创建多个）
            policy: policy.collect_mode
            tb_logger: TensorBoard logger
            exp_name: 实验名称
            instance_name: 实例名称
            policy_config: 策略配置，需要包含:
                - n_actors: Actor数量
                - envs_per_actor: 每个Actor管理的环境数量
            env_fn: 环境创建函数
            env_config: 环境配置
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
        self._n_actors = getattr(policy_config, 'n_actors', 4)  # 默认4个Actor
        self._envs_per_actor = getattr(policy_config, 'envs_per_actor', 8)  # 每个Actor 8个env
        
        # 保存环境创建信息
        self._env_fn = env_fn
        self._env_config = env_config
        self._original_env = env  # 保留原始env用于获取action_space等信息
        
        # 队列
        self._request_queue: queue.Queue = queue.Queue()
        self._response_queues: Dict[int, queue.Queue] = {}
        self._segment_queue: queue.Queue = queue.Queue()
        
        # 组件
        self._inference_server: Optional[GPUInferenceServer] = None
        self._actors: List[ActorWorker] = []
        self._actor_envs: List[BaseEnvManager] = []
        
        # 统计
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._episode_info = []
        
        # 数据池
        self.game_segment_pool = deque(maxlen=int(1e6))
        self.unroll_plus_td_steps = policy_config.num_unroll_steps + policy_config.td_steps
        
        self._logger.info(f"MultiActorMuZeroCollector initialized with {self._n_actors} actors, "
                         f"{self._envs_per_actor} envs per actor")
    
    def _create_actors(self):
        """创建多个Actor和对应的env_manager"""
        self._logger.info(f"Creating {self._n_actors} actors...")
        
        # 清理旧的actors
        for actor in self._actors:
            actor.stop()
        for env in self._actor_envs:
            env.close()
        self._actors.clear()
        self._actor_envs.clear()
        self._response_queues.clear()
        
        # 创建新的actors
        for actor_id in range(self._n_actors):
            # 为每个Actor创建独立的env_manager
            if self._env_fn is not None and self._env_config is not None:
                # 从env_config中取出该Actor对应的环境配置
                start_idx = actor_id * self._envs_per_actor
                end_idx = start_idx + self._envs_per_actor
                actor_env_configs = list(self._env_config[start_idx:end_idx])
                
                # 如果配置不够，循环使用
                while len(actor_env_configs) < self._envs_per_actor:
                    remaining = self._envs_per_actor - len(actor_env_configs)
                    actor_env_configs.extend(list(self._env_config[:remaining]))
                
                # 构造完整的env_manager配置（兼容不同版本的DI-engine）
                # 先定义所有可能需要的默认参数
                default_env_manager_cfg = dict(
                    type='subprocess',
                    shared_memory=False,
                    episode_num=float('inf'),
                    max_retry=5,
                    step_timeout=60,
                    auto_reset=True,
                    reset_timeout=60,
                    retry_type='reset',
                    retry_waiting_time=0.1,
                    copy_on_get=True,
                    context='spawn',
                    wait_num=float('inf'),
                    connect_timeout=60,
                )
                
                # 如果原始env有配置，用它覆盖默认值
                if hasattr(self._original_env, '_cfg') and self._original_env._cfg is not None:
                    # 从原始配置复制已有的值
                    for key, value in self._original_env._cfg.items():
                        default_env_manager_cfg[key] = value
                
                # 确保type字段存在
                default_env_manager_cfg['type'] = 'subprocess'
                
                env_manager_cfg = EasyDict(default_env_manager_cfg)
                
                actor_env = create_env_manager(
                    env_manager_cfg,
                    [partial(self._env_fn, cfg=c) for c in actor_env_configs]
                )
            else:
                # 复用原始env的设置
                actor_env = self._original_env
                if actor_id > 0:
                    self._logger.warning(f"Actor {actor_id}: Reusing original env_manager, "
                                        f"consider providing env_fn and env_config for true multi-actor")
            
            actor_env.seed(self.policy_config.seed + actor_id * 1000 if hasattr(self.policy_config, 'seed') else actor_id * 1000)
            actor_env.launch()
            self._actor_envs.append(actor_env)
            
            # 创建响应队列
            response_queue = queue.Queue()
            self._response_queues[actor_id] = response_queue
            
            # 创建Actor Worker
            actor = ActorWorker(
                actor_id=actor_id,
                env=actor_env,
                request_queue=self._request_queue,
                response_queue=response_queue,
                segment_queue=self._segment_queue,
                policy_config=self.policy_config,
                policy_reset_fn=self._policy.reset,
                logger=self._logger
            )
            self._actors.append(actor)
        
        self._logger.info(f"Created {len(self._actors)} actors successfully")
    
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
        
        # 停止推理服务器
        if self._inference_server:
            self._inference_server.stop()
        
        # 停止所有Actor
        for actor in self._actors:
            actor.stop()
        
        # 关闭所有环境
        for env in self._actor_envs:
            try:
                env.close()
            except:
                pass
        
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()
    
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
            return_data: [game_segments, meta_data] 与MuZeroCollector相同格式
        """
        if n_episode is None:
            n_episode = self._policy.get_attribute('cfg').get('n_episode', self._n_actors * self._envs_per_actor)
        
        if policy_kwargs is None:
            policy_kwargs = {'temperature': 1.0, 'epsilon': 0.0}
        
        temperature = policy_kwargs.get('temperature', 1.0)
        epsilon = policy_kwargs.get('epsilon', 0.0)
        
        self._logger.info(f"Starting multi-actor collect: n_episode={n_episode}, "
                         f"n_actors={self._n_actors}, temperature={temperature}, epsilon={epsilon}")
        
        start_time = time.time()
        
        # 创建Actors（如果尚未创建）
        if len(self._actors) == 0:
            self._create_actors()
        
        # 启动GPU推理服务器
        if self._inference_server is None:
            self._inference_server = GPUInferenceServer(
                policy=self._policy,
                request_queue=self._request_queue,
                response_queues=self._response_queues,
                policy_config=self.policy_config,
                logger=self._logger
            )
        self._inference_server.start()
        
        # 计算每个Actor需要收集的episode数量
        episodes_per_actor = n_episode // self._n_actors
        remainder = n_episode % self._n_actors
        
        # 启动所有Actor
        for i, actor in enumerate(self._actors):
            actor_episodes = episodes_per_actor + (1 if i < remainder else 0)
            actor.start(n_episode=actor_episodes, temperature=temperature, epsilon=epsilon)
        
        # 等待所有Actor完成并收集数据
        collected_segments = []
        total_collected = 0
        
        while total_collected < n_episode:
            # 检查Actor状态
            all_done = all(actor.is_done() for actor in self._actors)
            
            # 从segment队列收集数据
            while not self._segment_queue.empty():
                try:
                    segment: CollectedSegment = self._segment_queue.get_nowait()
                    collected_segments.append(segment)
                    total_collected += 1
                except queue.Empty:
                    break
            
            if all_done:
                break
            
            time.sleep(0.01)
        
        # 收集剩余数据
        while not self._segment_queue.empty():
            try:
                segment: CollectedSegment = self._segment_queue.get_nowait()
                collected_segments.append(segment)
            except queue.Empty:
                break
        
        # 停止所有Actor
        for actor in self._actors:
            actor.stop()
        
        # 停止推理服务器
        self._inference_server.stop()
        
        # 整理返回数据 - 与MuZeroCollector格式一致
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
        
        # 估算envstep（每个segment大约game_segment_length步）
        for segment in collected_segments:
            self._total_envstep_count += len(segment.game_segment.reward_segment)
        
        self._logger.info(f"Multi-actor collect finished: collected {len(collected_segments)} episodes "
                         f"in {collected_duration:.2f}s, total_envstep={self._total_envstep_count}")
        
        # 清空pool准备下次收集
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

