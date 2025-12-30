"""
中国象棋动作空间映射表

将所有可能的合法移动（from_square, to_square）映射到连续的动作索引。
这样可以将动作空间从 90*90=8100 压缩到约 2086 种实际可能的移动。

棋盘坐标系：
- 9列 (0-8)，10行 (0-9)
- square = row * 9 + col
- 红方在下方 (row 0-4)，黑方在上方 (row 5-9)

移动规则（只考虑几何规则，不考虑阻挡）：
- 车/炮: 横纵直线
- 马: 日字跳跃
- 象: 田字对角（不过河）
- 士: 九宫斜线
- 将: 九宫直线
- 兵: 前进（过河后可左右）
"""

from typing import Dict, List, Tuple, Set


def _square_to_coord(square: int) -> Tuple[int, int]:
    """格子编号转坐标 (row, col)"""
    return square // 9, square % 9


def _coord_to_square(row: int, col: int) -> int:
    """坐标转格子编号"""
    return row * 9 + col


def _is_valid_square(row: int, col: int) -> bool:
    """检查坐标是否在棋盘内"""
    return 0 <= row < 10 and 0 <= col < 9


def _generate_rook_cannon_moves() -> Set[Tuple[int, int]]:
    """
    生成车/炮的所有可能移动（横纵直线）
    车和炮的移动模式相同，只是吃子规则不同
    """
    moves = set()
    
    # 横向移动：同一行内任意两个不同的格子
    for row in range(10):
        for col1 in range(9):
            for col2 in range(9):
                if col1 != col2:
                    from_sq = _coord_to_square(row, col1)
                    to_sq = _coord_to_square(row, col2)
                    moves.add((from_sq, to_sq))
    
    # 纵向移动：同一列内任意两个不同的格子
    for col in range(9):
        for row1 in range(10):
            for row2 in range(10):
                if row1 != row2:
                    from_sq = _coord_to_square(row1, col)
                    to_sq = _coord_to_square(row2, col)
                    moves.add((from_sq, to_sq))
    
    return moves


def _generate_knight_moves() -> Set[Tuple[int, int]]:
    """
    生成马的所有可能移动（日字跳跃）
    8个方向：(±2, ±1) 和 (±1, ±2)
    """
    moves = set()
    deltas = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    
    for row in range(10):
        for col in range(9):
            from_sq = _coord_to_square(row, col)
            for dr, dc in deltas:
                new_row, new_col = row + dr, col + dc
                if _is_valid_square(new_row, new_col):
                    to_sq = _coord_to_square(new_row, new_col)
                    moves.add((from_sq, to_sq))
    
    return moves


def _generate_bishop_moves() -> Set[Tuple[int, int]]:
    """
    生成象的所有可能移动（田字对角，不过河）
    红方象只能在 row 0-4，黑方象只能在 row 5-9
    4个方向：(±2, ±2)
    """
    moves = set()
    deltas = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
    
    for row in range(10):
        for col in range(9):
            from_sq = _coord_to_square(row, col)
            for dr, dc in deltas:
                new_row, new_col = row + dr, col + dc
                if _is_valid_square(new_row, new_col):
                    # 检查是否过河：红方(row<=4)不能到row>4，黑方(row>=5)不能到row<5
                    # 简化处理：起点和终点必须在同一半场
                    from_side = 0 if row <= 4 else 1
                    to_side = 0 if new_row <= 4 else 1
                    if from_side == to_side:
                        to_sq = _coord_to_square(new_row, new_col)
                        moves.add((from_sq, to_sq))
    
    return moves


def _generate_advisor_moves() -> Set[Tuple[int, int]]:
    """
    生成士的所有可能移动（九宫内斜线）
    红方九宫：row 0-2, col 3-5
    黑方九宫：row 7-9, col 3-5
    4个方向：(±1, ±1)
    """
    moves = set()
    deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    # 红方九宫
    red_palace = [(r, c) for r in range(3) for c in range(3, 6)]
    # 黑方九宫
    black_palace = [(r, c) for r in range(7, 10) for c in range(3, 6)]
    
    for palace in [red_palace, black_palace]:
        palace_set = set(palace)
        for row, col in palace:
            from_sq = _coord_to_square(row, col)
            for dr, dc in deltas:
                new_row, new_col = row + dr, col + dc
                if (new_row, new_col) in palace_set:
                    to_sq = _coord_to_square(new_row, new_col)
                    moves.add((from_sq, to_sq))
    
    return moves


def _generate_king_moves() -> Set[Tuple[int, int]]:
    """
    生成将/帅的所有可能移动（九宫内直线）
    红方九宫：row 0-2, col 3-5
    黑方九宫：row 7-9, col 3-5
    4个方向：(±1, 0) 和 (0, ±1)
    """
    moves = set()
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # 红方九宫
    red_palace = [(r, c) for r in range(3) for c in range(3, 6)]
    # 黑方九宫
    black_palace = [(r, c) for r in range(7, 10) for c in range(3, 6)]
    
    for palace in [red_palace, black_palace]:
        palace_set = set(palace)
        for row, col in palace:
            from_sq = _coord_to_square(row, col)
            for dr, dc in deltas:
                new_row, new_col = row + dr, col + dc
                if (new_row, new_col) in palace_set:
                    to_sq = _coord_to_square(new_row, new_col)
                    moves.add((from_sq, to_sq))
    
    return moves


def _generate_pawn_moves() -> Set[Tuple[int, int]]:
    """
    生成兵/卒的所有可能移动
    红兵：初始在 row 3，前进方向 +1（向上）
    黑卒：初始在 row 6，前进方向 -1（向下）
    
    未过河时只能前进，过河后可前进或左右
    红兵过河：row >= 5
    黑卒过河：row <= 4
    """
    moves = set()
    
    # 红兵移动（row 3-9，前进方向+1）
    for row in range(3, 10):  # 红兵活动范围
        for col in range(9):
            from_sq = _coord_to_square(row, col)
            
            # 前进
            new_row = row + 1
            if new_row < 10:
                to_sq = _coord_to_square(new_row, col)
                moves.add((from_sq, to_sq))
            
            # 过河后（row >= 5）可以左右移动
            if row >= 5:
                for dc in [-1, 1]:
                    new_col = col + dc
                    if 0 <= new_col < 9:
                        to_sq = _coord_to_square(row, new_col)
                        moves.add((from_sq, to_sq))
    
    # 黑卒移动（row 0-6，前进方向-1）
    for row in range(0, 7):  # 黑卒活动范围
        for col in range(9):
            from_sq = _coord_to_square(row, col)
            
            # 前进
            new_row = row - 1
            if new_row >= 0:
                to_sq = _coord_to_square(new_row, col)
                moves.add((from_sq, to_sq))
            
            # 过河后（row <= 4）可以左右移动
            if row <= 4:
                for dc in [-1, 1]:
                    new_col = col + dc
                    if 0 <= new_col < 9:
                        to_sq = _coord_to_square(row, new_col)
                        moves.add((from_sq, to_sq))
    
    return moves


def _build_action_mapping() -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, int]]]:
    """
    构建动作映射表
    
    Returns:
        - MOVE_TO_ACTION: {(from_sq, to_sq): action_id}
        - ACTION_TO_MOVE: [(from_sq, to_sq), ...]，索引即为 action_id
    """
    # 收集所有可能的移动
    all_moves = set()
    all_moves.update(_generate_rook_cannon_moves())
    all_moves.update(_generate_knight_moves())
    all_moves.update(_generate_bishop_moves())
    all_moves.update(_generate_advisor_moves())
    all_moves.update(_generate_king_moves())
    all_moves.update(_generate_pawn_moves())
    
    # 排序以保证映射表稳定
    sorted_moves = sorted(all_moves)
    
    # 构建双向映射
    action_to_move = list(sorted_moves)
    move_to_action = {move: idx for idx, move in enumerate(sorted_moves)}
    
    return move_to_action, action_to_move


# ============================================================
# 全局映射表（模块加载时初始化）
# ============================================================

MOVE_TO_ACTION, ACTION_TO_MOVE = _build_action_mapping()
ACTION_SPACE_SIZE = len(ACTION_TO_MOVE)


def move_to_action(from_square: int, to_square: int) -> int:
    """
    将移动 (from_square, to_square) 转换为动作索引
    
    Args:
        from_square: 起始格子 (0-89)
        to_square: 目标格子 (0-89)
    
    Returns:
        动作索引
    
    Raises:
        KeyError: 如果移动不在合法移动表中
    """
    key = (from_square, to_square)
    if key not in MOVE_TO_ACTION:
        raise KeyError(f"移动 {key} 不在预定义的合法移动表中。"
                      f"from_square={from_square} ({_square_to_coord(from_square)}), "
                      f"to_square={to_square} ({_square_to_coord(to_square)})")
    return MOVE_TO_ACTION[key]


def action_to_move(action: int) -> Tuple[int, int]:
    """
    将动作索引转换为移动 (from_square, to_square)
    
    Args:
        action: 动作索引
    
    Returns:
        (from_square, to_square)
    
    Raises:
        IndexError: 如果动作索引超出范围
    """
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise IndexError(f"动作索引 {action} 超出范围 [0, {ACTION_SPACE_SIZE})")
    return ACTION_TO_MOVE[action]


def get_action_space_size() -> int:
    """获取动作空间大小"""
    return ACTION_SPACE_SIZE


# ============================================================
# 调试和统计信息
# ============================================================

def print_statistics():
    """打印动作空间统计信息"""
    print(f"动作空间大小: {ACTION_SPACE_SIZE}")
    print(f"车/炮直线移动: {len(_generate_rook_cannon_moves())}")
    print(f"马跳跃移动: {len(_generate_knight_moves())}")
    print(f"象田字移动: {len(_generate_bishop_moves())}")
    print(f"士斜线移动: {len(_generate_advisor_moves())}")
    print(f"将直线移动: {len(_generate_king_moves())}")
    print(f"兵卒移动: {len(_generate_pawn_moves())}")


if __name__ == "__main__":
    print_statistics()
    print(f"\n前10个动作映射:")
    for i in range(min(10, ACTION_SPACE_SIZE)):
        from_sq, to_sq = ACTION_TO_MOVE[i]
        from_coord = _square_to_coord(from_sq)
        to_coord = _square_to_coord(to_sq)
        print(f"  action {i}: {from_sq}({from_coord}) -> {to_sq}({to_coord})")
