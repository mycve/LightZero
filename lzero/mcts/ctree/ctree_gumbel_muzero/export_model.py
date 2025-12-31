"""
将 MuZero 模型导出为 TorchScript 格式

用于 C++ MCTS 搜索

使用方法:
    python export_model.py --checkpoint path/to/ckpt.pth --output model_recurrent.pt
"""

import argparse
import torch
import torch.nn as nn
from typing import NamedTuple


class NetworkOutput(NamedTuple):
    """模型输出结构"""
    latent_state: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    policy_logits: torch.Tensor


class RecurrentInferenceWrapper(nn.Module):
    """
    包装 recurrent_inference 以便导出为 TorchScript
    
    TorchScript 要求返回 tuple，不能返回 NamedTuple
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, latent_state: torch.Tensor, action: torch.Tensor):
        """
        执行 dynamics + prediction
        
        Args:
            latent_state: [batch, channels, height, width]
            action: [batch, 1] 
        
        Returns:
            tuple of (next_latent_state, reward, value, policy_logits)
        """
        # 调用模型的 recurrent_inference
        output = self.model.recurrent_inference(latent_state, action)
        
        # 返回 tuple（TorchScript 兼容）
        return (
            output.latent_state,
            output.reward,
            output.value,
            output.policy_logits
        )


def export_model(checkpoint_path: str, output_path: str, config_path: str = None):
    """
    导出模型为 TorchScript
    
    Args:
        checkpoint_path: 训练好的模型 checkpoint 路径
        output_path: 输出的 TorchScript 模型路径
        config_path: 可选，配置文件路径
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 这里需要根据实际的模型结构来创建模型
    # 示例：假设使用 MuZeroModel
    from lzero.model import MuZeroModel
    
    # 从 checkpoint 或 config 获取模型参数
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    elif config_path:
        import yaml
        with open(config_path) as f:
            model_config = yaml.safe_load(f)['model']
    else:
        raise ValueError("Need model config from checkpoint or config file")
    
    # 创建模型
    model = MuZeroModel(**model_config)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 创建包装器
    wrapper = RecurrentInferenceWrapper(model)
    wrapper.eval()
    
    # 创建示例输入用于 tracing
    batch_size = 1
    channels = model_config.get('image_channel', 68)  # 中国象棋默认 68
    height = 10
    width = 9
    
    example_latent_state = torch.randn(batch_size, channels, height, width)
    example_action = torch.randint(0, 2238, (batch_size, 1))  # 中国象棋动作空间
    
    print("Tracing model...")
    
    # 使用 torch.jit.trace 导出
    # 注意：如果模型有控制流，需要用 torch.jit.script
    try:
        traced_model = torch.jit.trace(
            wrapper, 
            (example_latent_state, example_action),
            strict=False
        )
    except Exception as e:
        print(f"Trace failed: {e}")
        print("Trying torch.jit.script...")
        traced_model = torch.jit.script(wrapper)
    
    # 保存
    traced_model.save(output_path)
    print(f"Model exported to {output_path}")
    
    # 验证
    print("Verifying exported model...")
    loaded_model = torch.jit.load(output_path)
    
    with torch.no_grad():
        original_output = wrapper(example_latent_state, example_action)
        loaded_output = loaded_model(example_latent_state, example_action)
        
        for i, (o1, o2) in enumerate(zip(original_output, loaded_output)):
            diff = (o1 - o2).abs().max().item()
            print(f"  Output {i} max diff: {diff:.6f}")
    
    print("Export successful!")


def export_for_chinese_chess(checkpoint_path: str, output_path: str):
    """
    专门为中国象棋导出模型
    """
    print("=" * 60)
    print("导出中国象棋 MuZero 模型")
    print("=" * 60)
    
    # 中国象棋的默认配置
    model_config = dict(
        model_type='conv',
        observation_shape=(68, 10, 9),
        action_space_size=2238,
        image_channel=68,
        num_res_blocks=9,
        num_channels=128,
    )
    
    from lzero.model import MuZeroModel
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型
    model = MuZeroModel(**model_config)
    
    # 尝试不同的 key 加载权重
    for key in ['model_state_dict', 'state_dict', 'model']:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            print(f"Loaded weights from checkpoint['{key}']")
            break
    else:
        # 可能整个 checkpoint 就是 state_dict
        try:
            model.load_state_dict(checkpoint)
            print("Loaded weights directly from checkpoint")
        except:
            raise ValueError("Cannot load model weights from checkpoint")
    
    model.eval()
    
    # 创建包装器并导出
    wrapper = RecurrentInferenceWrapper(model)
    wrapper.eval()
    
    # 示例输入
    example_latent = torch.randn(1, 68, 10, 9)
    example_action = torch.randint(0, 2238, (1, 1))
    
    print("Exporting...")
    traced = torch.jit.trace(wrapper, (example_latent, example_action), strict=False)
    traced.save(output_path)
    
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MuZero model to TorchScript")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", "-o", default="model_recurrent.pt", help="Output path")
    parser.add_argument("--config", help="Optional config file path")
    parser.add_argument("--chinese-chess", action="store_true", help="Use Chinese Chess defaults")
    
    args = parser.parse_args()
    
    if args.chinese_chess:
        export_for_chinese_chess(args.checkpoint, args.output)
    else:
        export_model(args.checkpoint, args.output, args.config)
