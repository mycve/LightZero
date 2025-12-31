"""
中国象棋 Gumbel MuZero 模型评估脚本

支持：
- 与 UCI 引擎（如 Pikafish）对战评估
- 人机对战
- HTML 回放生成

使用方法：
    python -m zoo.board_games.chinesechess.eval.cchess_gumbel_muzero_eval
"""

import numpy as np
from zoo.board_games.chinesechess.config.cchess_gumbel_muzero_config import main_config, create_config
from lzero.entry import eval_muzero


def evaluate_model(
    model_path: str,
    seeds: list = None,
    num_episodes_each_seed: int = 1,
    agent_vs_human: bool = False,
    render_mode: str = None,
    replay_path: str = './replay',
    uci_engine_path: str = None,
    engine_depth: int = 5,
):
    """
    评估 Gumbel MuZero 模型
    
    Args:
        model_path: 模型路径
        seeds: 随机种子列表
        num_episodes_each_seed: 每个种子运行的局数
        agent_vs_human: 是否与人对战
        render_mode: 渲染模式 ('human', 'html', None)
        replay_path: 回放保存路径
        uci_engine_path: UCI 引擎路径（如 'pikafish'）
        engine_depth: 引擎搜索深度
    """
    if seeds is None:
        seeds = [0]
    
    # 配置评估参数
    main_config.env.agent_vs_human = agent_vs_human
    main_config.env.render_mode = render_mode
    main_config.env.replay_path = replay_path
    
    # UCI 引擎配置（用于评估）
    if uci_engine_path:
        main_config.env.uci_engine_path = uci_engine_path
        main_config.env.engine_depth = engine_depth
    
    # 评估环境配置
    create_config.env_manager.type = 'base'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    
    total_test_episodes = num_episodes_each_seed * len(seeds)
    returns_mean_seeds = []
    returns_seeds = []
    
    print("=" * 60)
    print("中国象棋 Gumbel MuZero 模型评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"种子: {seeds}")
    print(f"每种子局数: {num_episodes_each_seed}")
    print(f"渲染模式: {render_mode}")
    print(f"UCI引擎: {uci_engine_path} (深度: {engine_depth})")
    print("=" * 60)
    
    for seed in seeds:
        print(f"\n>>> 评估种子 {seed}...")
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=True,
            model_path=model_path
        )
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)
    
    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("评估结果统计")
    print("=" * 60)
    print(f"总评估局数: {total_test_episodes}")
    print(f"各种子平均回报: {returns_mean_seeds}")
    print(f"总体平均回报: {returns_mean_seeds.mean():.4f}")
    
    wins = len(np.where(returns_seeds == 1.)[0])
    draws = len(np.where(returns_seeds == 0.)[0])
    losses = len(np.where(returns_seeds == -1.)[0])
    
    print(f"胜率: {wins}/{total_test_episodes} ({wins/total_test_episodes:.2%})")
    print(f"和率: {draws}/{total_test_episodes} ({draws/total_test_episodes:.2%})")
    print(f"负率: {losses}/{total_test_episodes} ({losses/total_test_episodes:.2%})")
    print("=" * 60)
    
    return returns_mean_seeds, returns_seeds


if __name__ == '__main__':
    # ============================================================
    # 评估配置
    # ============================================================
    
    # 模型路径（请修改为实际路径）
    MODEL_PATH = './data_gumbel_muzero/cchess_gumbel_muzero_sp-mode_ns50_upc50_seed0/ckpt/ckpt_best.pth.tar'
    
    # 评估参数
    SEEDS = [0]
    NUM_EPISODES = 1
    
    # 是否与人对战
    AGENT_VS_HUMAN = False
    
    # 渲染模式：None, 'human', 'html'
    RENDER_MODE = 'html'  # 生成 HTML 回放
    REPLAY_PATH = './replay_output'
    
    # UCI 引擎配置（用于作为对手）
    # 下载 Pikafish: https://github.com/official-pikafish/Pikafish
    UCI_ENGINE_PATH = None  # 例如: 'pikafish' 或 '/path/to/pikafish'
    ENGINE_DEPTH = 5
    
    # ============================================================
    # 执行评估
    # ============================================================
    
    evaluate_model(
        model_path=MODEL_PATH,
        seeds=SEEDS,
        num_episodes_each_seed=NUM_EPISODES,
        agent_vs_human=AGENT_VS_HUMAN,
        render_mode=RENDER_MODE,
        replay_path=REPLAY_PATH,
        uci_engine_path=UCI_ENGINE_PATH,
        engine_depth=ENGINE_DEPTH,
    )
