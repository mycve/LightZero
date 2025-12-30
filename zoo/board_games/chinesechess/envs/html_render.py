"""
中国象棋 HTML 回放生成器

生成可在浏览器中播放的 HTML 文件，支持：
- SVG 棋盘渲染
- 播放/暂停/快进/快退
- 步骤跳转
- 播放速度调节
"""

import json
from typing import List, Dict, Optional


# HTML 模板
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>中国象棋对局回放</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            color: #e0e0e0;
        }
        h1 {
            margin-bottom: 20px;
            color: #ffd700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .container {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            justify-content: center;
            max-width: 1200px;
        }
        .board-container {
            background: #2d2d44;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        #board-display {
            width: 420px;
            height: 470px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f0d9b5;
            border-radius: 10px;
        }
        #board-display svg {
            max-width: 100%;
            max-height: 100%;
        }
        .controls {
            background: #2d2d44;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            min-width: 300px;
        }
        .control-group {
            margin-bottom: 20px;
        }
        .control-group label {
            display: block;
            margin-bottom: 8px;
            color: #888;
            font-size: 14px;
        }
        .btn-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            padding: 12px 20px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #4a4a6a;
            color: #fff;
        }
        button:hover {
            background: #5a5a8a;
            transform: translateY(-2px);
        }
        button:active {
            transform: translateY(0);
        }
        button.primary {
            background: #ffd700;
            color: #1a1a2e;
            font-weight: bold;
        }
        button.primary:hover {
            background: #ffed4a;
        }
        .step-info {
            text-align: center;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .step-info .current-step {
            font-size: 32px;
            font-weight: bold;
            color: #ffd700;
        }
        .step-info .total-steps {
            font-size: 18px;
            color: #888;
        }
        .step-info .player {
            margin-top: 10px;
            font-size: 16px;
        }
        .player.red {
            color: #ff6b6b;
        }
        .player.black {
            color: #4ecdc4;
        }
        input[type="range"] {
            width: 100%;
            margin: 10px 0;
            -webkit-appearance: none;
            background: #1a1a2e;
            border-radius: 5px;
            height: 8px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #ffd700;
            border-radius: 50%;
            cursor: pointer;
        }
        .speed-display {
            text-align: center;
            color: #ffd700;
            font-weight: bold;
        }
        .move-list {
            max-height: 300px;
            overflow-y: auto;
            background: #1a1a2e;
            border-radius: 10px;
            padding: 10px;
        }
        .move-item {
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 5px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            transition: background 0.2s;
        }
        .move-item:hover {
            background: #3a3a5a;
        }
        .move-item.active {
            background: #4a4a6a;
            border-left: 3px solid #ffd700;
        }
        .move-item .move-num {
            color: #888;
            margin-right: 10px;
        }
        .move-item .move-text {
            flex: 1;
        }
        .keyboard-hint {
            margin-top: 20px;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 10px;
            font-size: 12px;
            color: #666;
        }
        .keyboard-hint kbd {
            background: #3a3a5a;
            padding: 2px 6px;
            border-radius: 3px;
            margin: 0 2px;
        }
    </style>
</head>
<body>
    <h1>中国象棋对局回放</h1>
    
    <div class="container">
        <div class="board-container">
            <div id="board-display">
                <!-- SVG 棋盘将插入这里 -->
            </div>
        </div>
        
        <div class="controls">
            <div class="step-info">
                <div>
                    <span class="current-step" id="current-step">0</span>
                    <span class="total-steps">/ <span id="total-steps">0</span></span>
                </div>
                <div class="player" id="player-info">初始局面</div>
            </div>
            
            <div class="control-group">
                <div class="btn-group">
                    <button onclick="gotoStart()" title="回到开始">⏮</button>
                    <button onclick="prevStep()" title="上一步">⏪</button>
                    <button onclick="togglePlay()" id="play-btn" class="primary" title="播放/暂停">▶</button>
                    <button onclick="nextStep()" title="下一步">⏩</button>
                    <button onclick="gotoEnd()" title="跳到结束">⏭</button>
                </div>
            </div>
            
            <div class="control-group">
                <label>播放速度</label>
                <input type="range" id="speed-slider" min="0.5" max="3" step="0.25" value="1" onchange="updateSpeed()">
                <div class="speed-display" id="speed-display">1.0x</div>
            </div>
            
            <div class="control-group">
                <label>进度</label>
                <input type="range" id="progress-slider" min="0" max="0" value="0" onchange="gotoStep(this.value)">
            </div>
            
            <div class="control-group">
                <label>走法列表</label>
                <div class="move-list" id="move-list">
                    <!-- 走法列表将动态生成 -->
                </div>
            </div>
            
            <div class="keyboard-hint">
                快捷键: <kbd>Space</kbd> 播放/暂停 | <kbd>←</kbd><kbd>→</kbd> 上一步/下一步 | <kbd>Home</kbd><kbd>End</kbd> 开始/结束
            </div>
        </div>
    </div>
    
    <script>
        // 棋盘帧数据
        const frames = FRAMES_DATA;
        const moves = MOVES_DATA;
        
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        let playSpeed = 1000; // 毫秒
        
        // 初始化
        function init() {
            document.getElementById('total-steps').textContent = frames.length - 1;
            document.getElementById('progress-slider').max = frames.length - 1;
            renderMoveList();
            updateDisplay();
        }
        
        // 渲染走法列表
        function renderMoveList() {
            const list = document.getElementById('move-list');
            list.innerHTML = '<div class="move-item active" onclick="gotoStep(0)"><span class="move-num">0</span><span class="move-text">初始局面</span></div>';
            
            moves.forEach((move, index) => {
                const playerText = move.player === 1 ? '红方' : '黑方';
                const fromCoord = squareToCoord(move.from);
                const toCoord = squareToCoord(move.to);
                const moveText = `${fromCoord} → ${toCoord}`;
                
                list.innerHTML += `<div class="move-item" onclick="gotoStep(${index + 1})">
                    <span class="move-num">${index + 1}</span>
                    <span class="move-text">${playerText}: ${moveText}</span>
                </div>`;
            });
        }
        
        // 格子坐标转换
        function squareToCoord(square) {
            const row = Math.floor(square / 9);
            const col = square % 9;
            const colLetter = String.fromCharCode(97 + col); // a-i
            return colLetter + row;
        }
        
        // 更新显示
        function updateDisplay() {
            document.getElementById('board-display').innerHTML = frames[currentStep];
            document.getElementById('current-step').textContent = currentStep;
            document.getElementById('progress-slider').value = currentStep;
            
            // 更新玩家信息
            const playerInfo = document.getElementById('player-info');
            if (currentStep === 0) {
                playerInfo.textContent = '初始局面';
                playerInfo.className = 'player';
            } else {
                const move = moves[currentStep - 1];
                if (move.player === 1) {
                    playerInfo.textContent = '红方走棋';
                    playerInfo.className = 'player red';
                } else {
                    playerInfo.textContent = '黑方走棋';
                    playerInfo.className = 'player black';
                }
            }
            
            // 更新走法列表高亮
            const items = document.querySelectorAll('.move-item');
            items.forEach((item, index) => {
                item.classList.toggle('active', index === currentStep);
            });
            
            // 滚动到当前项
            const activeItem = document.querySelector('.move-item.active');
            if (activeItem) {
                activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
        
        // 控制函数
        function nextStep() {
            if (currentStep < frames.length - 1) {
                currentStep++;
                updateDisplay();
            } else {
                stopPlay();
            }
        }
        
        function prevStep() {
            if (currentStep > 0) {
                currentStep--;
                updateDisplay();
            }
        }
        
        function gotoStart() {
            currentStep = 0;
            updateDisplay();
        }
        
        function gotoEnd() {
            currentStep = frames.length - 1;
            updateDisplay();
        }
        
        function gotoStep(step) {
            currentStep = parseInt(step);
            updateDisplay();
        }
        
        function togglePlay() {
            if (isPlaying) {
                stopPlay();
            } else {
                startPlay();
            }
        }
        
        function startPlay() {
            if (currentStep >= frames.length - 1) {
                currentStep = 0;
            }
            isPlaying = true;
            document.getElementById('play-btn').textContent = '⏸';
            playInterval = setInterval(() => {
                if (currentStep < frames.length - 1) {
                    currentStep++;
                    updateDisplay();
                } else {
                    stopPlay();
                }
            }, playSpeed);
        }
        
        function stopPlay() {
            isPlaying = false;
            document.getElementById('play-btn').textContent = '▶';
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
        }
        
        function updateSpeed() {
            const speedValue = parseFloat(document.getElementById('speed-slider').value);
            playSpeed = 1000 / speedValue;
            document.getElementById('speed-display').textContent = speedValue.toFixed(2) + 'x';
            
            if (isPlaying) {
                stopPlay();
                startPlay();
            }
        }
        
        // 键盘控制
        document.addEventListener('keydown', function(e) {
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    togglePlay();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    prevStep();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    nextStep();
                    break;
                case 'Home':
                    e.preventDefault();
                    gotoStart();
                    break;
                case 'End':
                    e.preventDefault();
                    gotoEnd();
                    break;
            }
        });
        
        // 初始化
        init();
    </script>
</body>
</html>
'''


def generate_html_replay(
    move_history: List[Dict],
    svg_frames: List[str],
    output_path: str
) -> None:
    """
    生成 HTML 回放文件
    
    Args:
        move_history: 移动历史列表，每项包含 {from, to, player, fen}
        svg_frames: SVG 帧列表
        output_path: 输出文件路径
    """
    # 如果没有帧，尝试从 move_history 生成
    if not svg_frames:
        svg_frames = ['<svg><text x="50%" y="50%" text-anchor="middle">无棋盘数据</text></svg>']
    
    # 确保帧数与步数匹配
    # 第一帧是初始局面，后续帧对应每一步
    while len(svg_frames) < len(move_history) + 1:
        svg_frames.append(svg_frames[-1] if svg_frames else '')
    
    # 转换数据为 JSON
    frames_json = json.dumps(svg_frames, ensure_ascii=False)
    moves_json = json.dumps(move_history, ensure_ascii=False)
    
    # 替换模板中的占位符
    html_content = HTML_TEMPLATE.replace('FRAMES_DATA', frames_json)
    html_content = html_content.replace('MOVES_DATA', moves_json)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_simple_html_replay(
    fen_history: List[str],
    output_path: str
) -> None:
    """
    从 FEN 历史生成简单的 HTML 回放（无需 SVG）
    
    Args:
        fen_history: FEN 字符串列表
        output_path: 输出文件路径
    """
    # 简化版本，使用文本显示
    frames = [f'<pre style="font-family: monospace; font-size: 14px;">{fen}</pre>' 
              for fen in fen_history]
    moves = [{'from': 0, 'to': 0, 'player': (i % 2) + 1} 
             for i in range(len(fen_history) - 1)]
    
    generate_html_replay(moves, frames, output_path)


if __name__ == '__main__':
    # 测试
    test_moves = [
        {'from': 60, 'to': 42, 'player': 1},
        {'from': 25, 'to': 43, 'player': 2},
    ]
    test_frames = [
        '<svg><rect width="100" height="100" fill="#f0d9b5"/><text x="50" y="50">初始</text></svg>',
        '<svg><rect width="100" height="100" fill="#f0d9b5"/><text x="50" y="50">第1步</text></svg>',
        '<svg><rect width="100" height="100" fill="#f0d9b5"/><text x="50" y="50">第2步</text></svg>',
    ]
    
    generate_html_replay(test_moves, test_frames, '/tmp/test_replay.html')
    print("测试 HTML 已生成: /tmp/test_replay.html")
