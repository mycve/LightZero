"""
中国象棋环境模块

主要组件：
- ChineseChessEnv: 中国象棋环境
- action_mapping: 动作映射（压缩动作空间）
- cchess: 中国象棋核心库
"""

from .cchess_env import ChineseChessEnv
from .action_mapping import (
    ACTION_SPACE_SIZE,
    move_to_action,
    action_to_move,
    MOVE_TO_ACTION,
    ACTION_TO_MOVE,
)
