"""
cchess库来自：https://github.com/windshadow233/python-chinese-chess/tree/main/cchess
修改了cchess库一处代码，提升位运算速度
2346行：def popcount(x: BitBoard) -> int:
    \"\"\"
    计算 BitBoard 中 1 的个数
    Python 3.10+ 原生 bit_count() 比 bin().count('1') 快 10+ 倍
    \"\"\"
    return x.bit_count()


pikafish引擎可以自行去下载：
https://github.com/official-pikafish/Pikafish
https://www.pikafish.com/

Overview:
    中国象棋环境，封装 cchess 库以适配 LightZero 的 BaseEnv 接口
    中国象棋是一个双人对弈游戏，棋盘为 9x10（9列10行）
    
    重构版本：
    - 动作空间从 8100 压缩到 2238（只保留合法移动模式）
    - 观察空间去除颜色层，使用己方优先编码
    - 简化环境拷贝，使用 deepcopy
    - 支持 HTML 回放
    
Mode:
    - ``self_play_mode``: 自对弈模式，用于 AlphaZero/MuZero 数据生成
    - ``play_with_bot_mode``: 与内置 bot 对战模式
    - ``eval_mode``: 评估模式
"""

import copy
import os
from typing import List, Any, Tuple, Optional
from collections import deque

import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict
from gymnasium import spaces

from . import cchess
from .action_mapping import (
    move_to_action as _move_to_action,
    action_to_move as _action_to_move,
    ACTION_SPACE_SIZE,
    MOVE_TO_ACTION,
)


def move_to_action(move: cchess.Move) -> int:
    """将 Move 对象转换为动作索引（使用压缩映射）"""
    return _move_to_action(move.from_square, move.to_square)


def action_to_move(action: int) -> cchess.Move:
    """将动作索引转换为 Move 对象（使用压缩映射）"""
    from_square, to_square = _action_to_move(action)
    return cchess.Move(from_square, to_square)


@ENV_REGISTRY.register('cchess')
class ChineseChessEnv(BaseEnv):
    """
    中国象棋环境（重构版）
    
    主要特性：
    - 动作空间：2238（压缩后的合法移动）
    - 观察空间：(56, 10, 9) = 14层棋子 * 4历史帧，己方优先编码
    - 固定视角：始终从当前行动方视角观察，己方棋子在前7层
    """
    
    config = dict(
        env_id="ChineseChess",
        battle_mode='self_play_mode',
        battle_mode_in_simulation_env='self_play_mode',
        render_mode=None,  # 'human', 'svg', 'html'
        replay_path=None,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        uci_engine_path=None,  # UCI引擎路径，如 'pikafish' 或 '/path/to/pikafish'
        engine_depth=5,  # 引擎搜索深度，通常1-20，深度越大越强
        channel_last=False,
        scale=False,
        stop_value=2,
        max_episode_steps=500,  # 最大回合数限制，防止无限回合
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None) -> None:
        self.cfg = cfg
        self.channel_last = cfg.channel_last
        self.scale = cfg.scale
        
        self.render_mode = cfg.render_mode
        self.replay_path = cfg.replay_path
        
        self.battle_mode = cfg.battle_mode
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        self.battle_mode_in_simulation_env = 'self_play_mode'
        
        self.agent_vs_human = cfg.agent_vs_human
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        
        # UCI引擎配置
        self.uci_engine_path = cfg.get('uci_engine_path', None)
        self.engine_depth = cfg.get('engine_depth', 5)
        self.engine = None
        
        # 初始化UCI引擎（如果配置了）
        if self.uci_engine_path:
            try:
                from .cchess import engine
                self.engine = engine.SimpleEngine.popen_uci(self.uci_engine_path)
                logging.info(f"UCI引擎加载成功: {self.uci_engine_path}")
            except Exception as e:
                logging.warning(f"UCI引擎加载失败: {e}，将使用随机策略")
                self.engine = None
        
        # 最大步数限制
        self.max_episode_steps = cfg.max_episode_steps
        self.current_step = 0
        
        # 渲染相关
        self.frames = []  # 用于保存渲染帧
        self.move_history = []  # 用于 HTML 回放
        
        # 初始化棋盘
        self.board = cchess.Board()
        
        self.players = [1, 2]  # 1: 红方(RED), 2: 黑方(BLACK)
        self._current_player = 1
        self._env = self

        # 历史观测堆叠
        self.stack_obs_num = 4
        self.obs_buffer = deque(maxlen=self.stack_obs_num)

        # 预计算：Board 棋子遍历所需的查找表
        self._piece_types = [cchess.PAWN, cchess.ROOK, cchess.KNIGHT, cchess.CANNON, 
                             cchess.ADVISOR, cchess.BISHOP, cchess.KING]
        
        # 预计算：BitBoard位索引到(row, col)的映射
        self._square_to_coord = np.array([(s // 9, s % 9) for s in range(90)], dtype=np.int32)

    def _get_pieces_planes(self, color: bool) -> np.ndarray:
        """
        获取指定颜色的棋子平面表示
        
        Args:
            color: cchess.RED 或 cchess.BLACK
            
        Returns:
            shape (7, 10, 9) 的数组，每层对应一种棋子类型
        """
        planes = np.zeros((7, 10, 9), dtype=np.float32)
        for i, piece_type in enumerate(self._piece_types):
            mask = self.board.pieces_mask(piece_type, color)
            if mask:
                for square in cchess.scan_forward(mask):
                    r, c = self._square_to_coord[square]
                    planes[i, r, c] = 1
        return planes

    def _get_canonical_planes(self) -> np.ndarray:
        """
        获取己方优先编码的棋盘表示（固定视角）
        
        始终将当前行动方的棋子放在前7层，对手棋子放在后7层。
        不再旋转棋盘，也不再添加颜色层。
        
        Returns:
            shape (14, 10, 9) 的数组
        """
        if self._current_player == 1:  # 红方行动
            own_planes = self._get_pieces_planes(cchess.RED)
            opp_planes = self._get_pieces_planes(cchess.BLACK)
        else:  # 黑方行动
            own_planes = self._get_pieces_planes(cchess.BLACK)
            opp_planes = self._get_pieces_planes(cchess.RED)
        
        return np.concatenate([own_planes, opp_planes], axis=0)

    def _update_obs_buffer(self):
        """更新观测缓存"""
        planes = self._get_canonical_planes()
        self.obs_buffer.append(planes)

    def _player_step(self, action: int, flag: str) -> BaseEnvTimestep:
        """
        执行一步棋
        """
        legal_actions = self.legal_actions
        
        if action not in legal_actions:
            logging.warning(
                f"非法动作: {action}, 合法动作有 {len(legal_actions)} 个。"
                f"标志: {flag}. 随机选择一个合法动作。"
            )
            action = self.random_action()
        
        # 保存执行动作的玩家（用于奖励计算）
        acting_player = self._current_player
        
        move = action_to_move(action)
        
        # 记录移动历史（用于 HTML 回放）
        self.move_history.append({
            'from': move.from_square,
            'to': move.to_square,
            'player': acting_player,
            'fen': self.board.fen()
        })
        
        self.board.push(move)
        
        # 增加步数计数
        self.current_step += 1
        
        # board.push() 已经自动切换了 turn，需要同步更新 _current_player
        self._current_player = 1 if self.board.turn else 2

        # 更新观测历史
        self._update_obs_buffer()
        
        # 检查游戏是否结束
        done = self.board.is_game_over()
        outcome = self.board.outcome()
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_episode_steps:
            done = True
            outcome = None  # 达到最大步数视为平局
        
        if done:
            # [DEBUG] 详细打印游戏结束原因
            termination_reason = outcome.termination if outcome else "MaxSteps/Unknown"
            winner_info = "None"
            if outcome and outcome.winner is not None:
                winner_info = "RED" if outcome.winner == cchess.RED else "BLACK"
            
            if outcome and outcome.winner is not None:
                # 有明确的胜者，奖励从执行动作的玩家视角计算
                if outcome.winner == cchess.RED:
                    reward_scalar = 1.0 if acting_player == 1 else -1.0
                else:
                    reward_scalar = -1.0 if acting_player == 1 else 1.0
                logging.info(f"[ENV] Game Won! Winner: {winner_info}, ActingPlayer: {acting_player}, "
                           f"Reward: {reward_scalar}, Reason: {termination_reason}, Steps: {self.current_step}")
            else:
                # 和棋或特殊情况处理
                if termination_reason == cchess.Termination.FOURFOLD_REPETITION:
                    reward_scalar = -1.0  # 重复局面判负
                    logging.info(f"[ENV] Repetition! ActingPlayer: {acting_player} LOSE, Steps: {self.current_step}")
                elif self.current_step >= self.max_episode_steps:
                    reward_scalar = -1.0  # 超时判负
                    logging.info(f"[ENV] MaxSteps! ActingPlayer: {acting_player} LOSE, Steps: {self.current_step}")
                else:
                    reward_scalar = 0.0
                    logging.info(f"[ENV] Draw. Reason: {termination_reason}, Steps: {self.current_step}")
        else:
            reward_scalar = 0.0
        
        reward = np.array([reward_scalar], dtype=np.float32)
        info = {}
        obs = self.observe()
        
        return BaseEnvTimestep(obs, reward, done, info)

    def step(self, action: int) -> BaseEnvTimestep:
        """环境的 step 函数"""
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action()
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.random_action()
            
            timestep = self._player_step(action, "agent")
            
            if timestep.done:
                reward_scalar = float(timestep.reward[0])
                timestep.info['eval_episode_return'] = reward_scalar
            
            return timestep
        
        elif self.battle_mode == 'play_with_bot_mode':
            timestep_player1 = self._player_step(action, "bot_agent")
            
            if timestep_player1.done:
                timestep_player1.info['eval_episode_return'] = float(timestep_player1.reward[0])
                timestep_player1.obs['to_play'] = np.array([-1], dtype=np.int32)
                return timestep_player1
            
            bot_action = self.bot_action()
            timestep_player2 = self._player_step(bot_action, "bot_bot")
            
            reward_scalar = float(timestep_player2.reward[0])
            timestep_player2.info['eval_episode_return'] = -reward_scalar
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)
            timestep_player2.obs['to_play'] = np.array([1], dtype=np.int32)
            
            return timestep_player2
        
        elif self.battle_mode == 'eval_mode':
            timestep_player1 = self._player_step(action, "eval_agent")
            
            if timestep_player1.done:
                timestep_player1.info['eval_episode_return'] = float(timestep_player1.reward[0])
                timestep_player1.obs['to_play'] = np.array([-1], dtype=np.int32)
                return timestep_player1
            
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()
            
            timestep_player2 = self._player_step(bot_action, "eval_bot")
            
            reward_scalar = float(timestep_player2.reward[0])
            timestep_player2.info['eval_episode_return'] = -reward_scalar
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)
            timestep_player2.obs['to_play'] = np.array([1], dtype=np.int32)
            
            return timestep_player2

    def reset(self, start_player_index: int = 0, init_state: Optional[str] = None) -> dict:
        """重置环境"""
        if init_state is None:
            self.board = cchess.Board()
        else:
            self.board = cchess.Board(fen=init_state)
        
        self.players = [1, 2]
        self.start_player_index = start_player_index
        self.current_step = 0
        self.frames = []
        self.move_history = []
        
        self._current_player = 1 if self.board.turn else 2

        # 重置历史观测
        self.obs_buffer.clear()
        init_planes = self._get_canonical_planes()
        for _ in range(self.stack_obs_num):
            self.obs_buffer.append(init_planes)
        
        # 设置动作空间和观察空间（使用压缩后的动作空间）
        self._action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self._reward_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 观察空间：(14 * stack, 10, 9) - 去除颜色层
        obs_channels = 14 * self.stack_obs_num  # 56
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0, high=1, shape=(obs_channels, 10, 9), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8),
                "board": spaces.Box(low=0, high=7, shape=(10, 9), dtype=np.int8),
                "current_player_index": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
                "to_play": spaces.Box(low=-1, high=2, shape=(1,), dtype=np.int32),
            }
        )
        
        return self.observe()

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前堆叠状态（己方优先编码，无颜色层）
        """
        # 堆叠历史帧，shape: (14 * stack, 10, 9) = (56, 10, 9)
        state = np.concatenate(list(self.obs_buffer), axis=0)
        
        if self.scale:
            scale_state = state / 2
        else:
            scale_state = state
        
        if self.channel_last:
            return np.transpose(state, [1, 2, 0]), np.transpose(scale_state, [1, 2, 0])
        else:
            return state, scale_state

    def observe(self) -> dict:
        """返回观察"""
        legal_actions_list = self.legal_actions
        
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        for action in legal_actions_list:
            action_mask[action] = 1
        
        # 棋盘可视化表示
        board_visual = np.zeros((10, 9), dtype=np.int8)
        for square in range(90):
            piece = self.board.piece_at(square)
            if piece:
                row = cchess.square_row(square)
                col = cchess.square_column(square)
                board_visual[row, col] = piece.piece_type
        
        if self.battle_mode in ['play_with_bot_mode', 'eval_mode']:
            return {
                "observation": self.current_state()[1],
                "action_mask": action_mask,
                "board": board_visual,
                "current_player_index": np.array([self.players.index(self._current_player)], dtype=np.int32),
                "to_play": np.array([-1], dtype=np.int32)
            }
        else:  # self_play_mode
            return {
                "observation": self.current_state()[1],
                "action_mask": action_mask,
                "board": board_visual,
                "current_player_index": np.array([self.players.index(self._current_player)], dtype=np.int32),
                "to_play": np.array([self._current_player], dtype=np.int32)
            }

    @property
    def legal_actions(self) -> List[int]:
        """返回所有合法动作的索引列表（使用压缩映射）"""
        legal_actions_list = []
        for move in self.board.legal_moves:
            key = (move.from_square, move.to_square)
            if key in MOVE_TO_ACTION:
                legal_actions_list.append(MOVE_TO_ACTION[key])
            else:
                # 这不应该发生，但为了安全起见记录警告
                logging.warning(f"移动 {key} 不在映射表中，跳过")
        return legal_actions_list

    def get_done_winner(self) -> Tuple[bool, int]:
        """检查游戏是否结束并返回胜者"""
        if self.current_step >= self.max_episode_steps:
            return True, -1
        
        done = self.board.is_game_over()
        if not done:
            return False, -1
        
        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            return True, -1
        elif outcome.winner == cchess.RED:
            return True, 1
        else:
            return True, 2

    def get_done_reward(self) -> Tuple[bool, Optional[int]]:
        """检查游戏是否结束并从玩家1的视角返回奖励"""
        done, winner = self.get_done_winner()
        if not done:
            return False, None
        
        if winner == 1:
            return True, 1
        elif winner == 2:
            return True, -1
        else:
            return True, 0

    def random_action(self) -> int:
        """随机选择一个合法动作"""
        return np.random.choice(self.legal_actions)
    
    def bot_action(self) -> int:
        """使用UCI引擎或随机策略选择动作"""
        if self.engine is not None:
            try:
                from .cchess import engine as engine_module
                limit = engine_module.Limit(depth=self.engine_depth)
                result = self.engine.play(self.board, limit)
                return move_to_action(result.move)
            except Exception as e:
                logging.warning(f"引擎调用失败: {e}，使用随机策略")
                return self.random_action()
        else:
            return self.random_action()

    def human_to_action(self) -> int:
        """从人类输入获取动作"""
        print(self.board.unicode(axes=True, axes_type=0))
        while True:
            try:
                uci = input(f"请输入走法（UCI格式，如 h2e2）: ")
                move = cchess.Move.from_uci(uci)
                action = move_to_action(move)
                if action in self.legal_actions:
                    return action
                else:
                    print("非法走法，请重新输入")
            except KeyboardInterrupt:
                print("退出")
                import sys
                sys.exit(0)
            except Exception as e:
                print(f"输入错误: {e}，请重新输入")

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return "LightZero ChineseChess Env (Refactored)"

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def current_player_index(self) -> int:
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self) -> int:
        return self.players[0] if self._current_player == self.players[1] else self.players[1]

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space

    def copy(self) -> 'ChineseChessEnv':
        """复制环境（使用 deepcopy 简化实现）"""
        return copy.deepcopy(self)

    def simulate_action(self, action: int) -> Any:
        """模拟执行动作并返回新的模拟环境（用于 MCTS）"""
        if action not in self.legal_actions:
            raise ValueError(f"动作 {action} 不合法")
        
        new_env = self.copy()
        move = action_to_move(action)
        new_env.board.push(move)
        new_env.current_step += 1
        new_env._current_player = 1 if new_env.board.turn else 2
        new_env._update_obs_buffer()
        
        return new_env

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def render(self, mode: str = None) -> Optional[str]:
        """
        渲染棋盘
        
        Args:
            mode: 渲染模式
                - 'state_realtime_mode' / 'human': 打印棋盘到控制台
                - 'image_savefile_mode': 保存 SVG 帧
                - 'svg': 返回 SVG 字符串
                - 'html': 保存 HTML 回放文件（游戏结束时）
        """
        mode = mode or self.render_mode
        
        if mode is None:
            return None
        
        if mode in ['state_realtime_mode', 'human']:
            print("\n" + "=" * 50)
            print(f"步数: {self.current_step} | 当前玩家: {'红方' if self._current_player == 1 else '黑方'}")
            print(self.board.unicode(axes=True, axes_type=1))
            print("=" * 50)
            return None
        
        elif mode == 'image_savefile_mode':
            try:
                from .cchess import svg
                last_move = self.board.peek() if self.board.move_stack else None
                svg_str = svg.board(self.board, lastmove=last_move, size=400)
                self.frames.append(svg_str)
            except Exception as e:
                logging.warning(f"SVG渲染失败: {e}")
            return None
        
        elif mode == 'svg':
            try:
                from .cchess import svg
                last_move = self.board.peek() if self.board.move_stack else None
                return svg.board(self.board, lastmove=last_move, size=400)
            except Exception as e:
                logging.warning(f"SVG渲染失败: {e}")
                return None
        
        elif mode == 'html':
            # HTML 模式：记录帧用于后续生成 HTML
            try:
                from .cchess import svg
                last_move = self.board.peek() if self.board.move_stack else None
                svg_str = svg.board(self.board, lastmove=last_move, size=400)
                self.frames.append(svg_str)
            except Exception as e:
                logging.warning(f"SVG渲染失败: {e}")
            return None
        
        else:
            logging.warning(f"不支持的渲染模式: {mode}")
            return None
    
    def save_render_output(self, replay_path: str = None, format: str = 'svg') -> Optional[str]:
        """
        保存渲染输出到文件
        
        Args:
            replay_path: 保存路径
            format: 'svg' 或 'html'
            
        Returns:
            HTML 格式时返回文件路径
        """
        if not self.frames and format != 'html':
            logging.warning("没有可保存的渲染帧")
            return None
        
        save_path = replay_path or self.replay_path
        if save_path is None:
            save_path = './replay_output'
        
        os.makedirs(save_path, exist_ok=True)
        
        if format == 'svg':
            for i, svg_str in enumerate(self.frames):
                file_path = os.path.join(save_path, f'step_{i:04d}.svg')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(svg_str)
            logging.info(f"已保存 {len(self.frames)} 个SVG文件到 {save_path}")
            self.frames = []
            return None
        
        elif format == 'html':
            from .html_render import generate_html_replay
            html_path = os.path.join(save_path, 'replay.html')
            generate_html_replay(self.move_history, self.frames, html_path)
            logging.info(f"已保存 HTML 回放到 {html_path}")
            self.frames = []
            return html_path
        
        else:
            logging.warning(f"不支持的保存格式: {format}")
            return None
    
    def close(self) -> None:
        """关闭环境，释放资源"""
        if self.engine is not None:
            try:
                self.engine.quit()
                logging.info("UCI引擎已关闭")
            except Exception as e:
                logging.warning(f"关闭引擎时出错: {e}")
            finally:
                self.engine = None
