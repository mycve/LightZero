/**
 * 完全 C++ 化的 MCTS 搜索（带模型推理）
 * 
 * 设计思路：
 * 1. 使用 libtorch 在 C++ 中进行模型推理
 * 2. 整个 search 循环在 C++ 中完成，消除 Python GIL
 * 3. 通过 TorchScript 加载预训练模型
 * 
 * 依赖：
 * - libtorch (PyTorch C++ 前端)
 * - CUDA (可选，GPU 加速)
 * 
 * 编译方式：见 CMakeLists.txt
 */

#ifndef CNODE_WITH_MODEL_H
#define CNODE_WITH_MODEL_H

#include "cnode.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>

namespace tree {

/**
 * 模型输出结构
 */
struct NetworkOutput {
    torch::Tensor latent_state;   // 隐状态
    torch::Tensor reward;         // 奖励
    torch::Tensor value;          // 价值
    torch::Tensor policy_logits;  // 策略 logits
};

/**
 * C++ MCTS 搜索器（带模型推理）
 */
class CMCTSSearcher {
public:
    CMCTSSearcher(
        const std::string& model_path,  // TorchScript 模型路径
        int num_simulations,
        int max_num_considered_actions,
        float discount_factor,
        float value_delta_max,
        bool use_cuda = true
    );
    
    ~CMCTSSearcher();
    
    /**
     * 完整的 MCTS 搜索（完全在 C++ 中执行）
     * 
     * @param roots          根节点批次
     * @param latent_states  初始隐状态 [batch, channels, height, width]
     * @param to_play_batch  当前玩家列表
     */
    void search(
        CRoots* roots,
        const torch::Tensor& latent_states,
        const std::vector<int>& to_play_batch
    );
    
private:
    // TorchScript 模型
    torch::jit::script::Module model_;
    
    // 配置参数
    int num_simulations_;
    int max_num_considered_actions_;
    float discount_factor_;
    float value_delta_max_;
    
    // 设备
    torch::Device device_;
    
    // 值支持范围（用于逆变换）
    float value_support_min_;
    float value_support_max_;
    float reward_support_min_;
    float reward_support_max_;
    
    /**
     * 模型推理：recurrent_inference
     */
    NetworkOutput recurrent_inference(
        const torch::Tensor& latent_states,
        const torch::Tensor& actions
    );
    
    /**
     * 逆标量变换
     */
    torch::Tensor inverse_scalar_transform(
        const torch::Tensor& logits,
        float support_min,
        float support_max
    );
};

/**
 * Python 绑定的包装函数
 * 
 * 在 Cython/pybind11 中调用这个函数
 */
void cbatch_search_with_model(
    CRoots* roots,
    const std::string& model_path,
    float* latent_states_data,  // numpy 数组指针
    int batch_size,
    int channels,
    int height,
    int width,
    int* to_play_batch,
    int num_simulations,
    int max_num_considered_actions,
    float discount_factor,
    float value_delta_max,
    bool use_cuda
);

}  // namespace tree

#endif  // CNODE_WITH_MODEL_H
