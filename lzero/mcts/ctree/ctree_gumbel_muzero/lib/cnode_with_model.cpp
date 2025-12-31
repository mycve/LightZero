/**
 * 完全 C++ 化的 MCTS 搜索实现
 */

#include "cnode_with_model.h"
#include <iostream>
#include <chrono>

namespace tree {

CMCTSSearcher::CMCTSSearcher(
    const std::string& model_path,
    int num_simulations,
    int max_num_considered_actions,
    float discount_factor,
    float value_delta_max,
    bool use_cuda
) : num_simulations_(num_simulations),
    max_num_considered_actions_(max_num_considered_actions),
    discount_factor_(discount_factor),
    value_delta_max_(value_delta_max),
    device_(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    try {
        // 加载 TorchScript 模型
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
        
        std::cout << "[CMCTSSearcher] Model loaded successfully on " 
                  << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[CMCTSSearcher] Error loading model: " << e.what() << std::endl;
        throw;
    }
    
    // 默认支持范围（需要从配置中读取）
    value_support_min_ = -2.0;
    value_support_max_ = 3.0;
    reward_support_min_ = -2.0;
    reward_support_max_ = 3.0;
}

CMCTSSearcher::~CMCTSSearcher() {}

torch::Tensor CMCTSSearcher::inverse_scalar_transform(
    const torch::Tensor& logits,
    float support_min,
    float support_max
) {
    /**
     * 逆标量变换：将分类分布转换回标量值
     * 
     * 简化版本，实际需要根据 LightZero 的实现调整
     */
    auto probs = torch::softmax(logits, -1);
    int support_size = logits.size(-1);
    
    // 创建支持值
    auto support = torch::linspace(support_min, support_max, support_size, 
                                   torch::TensorOptions().device(device_));
    
    // 期望值
    auto value = (probs * support).sum(-1);
    
    return value;
}

NetworkOutput CMCTSSearcher::recurrent_inference(
    const torch::Tensor& latent_states,
    const torch::Tensor& actions
) {
    /**
     * 递归推理：dynamics + prediction
     * 
     * 输入：
     *   - latent_states: [batch, channels, height, width]
     *   - actions: [batch, 1]
     * 
     * 输出：
     *   - next_latent_state, reward, value, policy_logits
     */
    torch::NoGradGuard no_grad;
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(latent_states);
    inputs.push_back(actions);
    
    // 调用模型的 recurrent_inference 方法
    // 注意：需要模型导出时包含这个方法
    auto output = model_.forward(inputs);
    
    // 解析输出（根据模型实际输出格式调整）
    auto output_tuple = output.toTuple();
    
    NetworkOutput result;
    result.latent_state = output_tuple->elements()[0].toTensor();
    result.reward = output_tuple->elements()[1].toTensor();
    result.value = output_tuple->elements()[2].toTensor();
    result.policy_logits = output_tuple->elements()[3].toTensor();
    
    return result;
}

void CMCTSSearcher::search(
    CRoots* roots,
    const torch::Tensor& latent_states_input,
    const std::vector<int>& to_play_batch
) {
    /**
     * 完整的 MCTS 搜索循环
     * 
     * 整个过程在 C++ 中完成，没有 Python GIL 开销
     */
    torch::NoGradGuard no_grad;
    
    int batch_size = roots->root_num;
    
    // 将输入移到正确的设备
    auto latent_states = latent_states_input.to(device_);
    
    // 存储所有搜索路径中的 latent states
    std::vector<torch::Tensor> latent_state_batch_in_search_path;
    latent_state_batch_in_search_path.push_back(latent_states);
    
    // MinMax 统计
    tools::CMinMaxStatsList min_max_stats_lst(batch_size);
    min_max_stats_lst.set_delta(value_delta_max_);
    
    // to_play 副本
    std::vector<int> virtual_to_play_batch = to_play_batch;
    
    // ============================================
    // 主搜索循环 - 完全在 C++ 中执行
    // ============================================
    for (int simulation_index = 0; simulation_index < num_simulations_; ++simulation_index) {
        
        // 结果包装器
        CSearchResults results(batch_size);
        
        // 重置 virtual_to_play_batch
        virtual_to_play_batch = to_play_batch;
        
        // ============================================
        // 阶段 1: 选择 (batch_traverse)
        // ============================================
        cbatch_traverse(roots, num_simulations_, max_num_considered_actions_,
                       discount_factor_, results, virtual_to_play_batch);
        
        // 收集叶子节点的 latent states
        std::vector<torch::Tensor> leaf_latent_states;
        leaf_latent_states.reserve(batch_size);
        
        for (int i = 0; i < batch_size; ++i) {
            int ix = results.latent_state_index_in_search_path[i];
            int iy = results.latent_state_index_in_batch[i];
            leaf_latent_states.push_back(latent_state_batch_in_search_path[ix][iy]);
        }
        
        // 堆叠成批次
        auto leaf_states_tensor = torch::stack(leaf_latent_states, 0);
        
        // 构建 actions tensor
        auto actions_tensor = torch::from_blob(
            results.last_actions.data(),
            {batch_size, 1},
            torch::TensorOptions().dtype(torch::kInt32)
        ).to(device_).to(torch::kLong);
        
        // ============================================
        // 阶段 2: 扩展 (recurrent_inference)
        // ============================================
        NetworkOutput network_output = recurrent_inference(leaf_states_tensor, actions_tensor);
        
        // 逆标量变换
        auto values = inverse_scalar_transform(network_output.value, 
                                               value_support_min_, value_support_max_);
        auto rewards = inverse_scalar_transform(network_output.reward,
                                                reward_support_min_, reward_support_max_);
        
        // 转到 CPU 并转换为 vector
        auto latent_state_cpu = network_output.latent_state.cpu();
        auto policy_logits_cpu = network_output.policy_logits.cpu();
        auto values_cpu = values.cpu();
        auto rewards_cpu = rewards.cpu();
        
        // 存储新的 latent states
        latent_state_batch_in_search_path.push_back(latent_state_cpu);
        
        // 转换为 C++ vector
        std::vector<float> reward_batch(rewards_cpu.data_ptr<float>(),
                                        rewards_cpu.data_ptr<float>() + batch_size);
        std::vector<float> value_batch(values_cpu.data_ptr<float>(),
                                       values_cpu.data_ptr<float>() + batch_size);
        
        // policy_logits 转换为 2D vector
        std::vector<std::vector<float>> policy_logits_batch(batch_size);
        auto policy_accessor = policy_logits_cpu.accessor<float, 2>();
        for (int i = 0; i < batch_size; ++i) {
            int action_size = policy_logits_cpu.size(1);
            policy_logits_batch[i].resize(action_size);
            for (int j = 0; j < action_size; ++j) {
                policy_logits_batch[i][j] = policy_accessor[i][j];
            }
        }
        
        // ============================================
        // 阶段 3: 回溯 (batch_back_propagate)
        // ============================================
        int current_latent_state_index = simulation_index + 1;
        cbatch_back_propagate(
            current_latent_state_index,
            discount_factor_,
            reward_batch,
            value_batch,
            policy_logits_batch,
            &min_max_stats_lst,
            results,
            virtual_to_play_batch
        );
    }
}

// ============================================
// Python 绑定的包装函数
// ============================================
void cbatch_search_with_model(
    CRoots* roots,
    const std::string& model_path,
    float* latent_states_data,
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
) {
    // 创建搜索器
    CMCTSSearcher searcher(
        model_path,
        num_simulations,
        max_num_considered_actions,
        discount_factor,
        value_delta_max,
        use_cuda
    );
    
    // 从原始数据创建 tensor
    auto latent_states = torch::from_blob(
        latent_states_data,
        {batch_size, channels, height, width},
        torch::TensorOptions().dtype(torch::kFloat32)
    );
    
    // 创建 to_play vector
    std::vector<int> to_play_vec(to_play_batch, to_play_batch + batch_size);
    
    // 执行搜索
    searcher.search(roots, latent_states, to_play_vec);
}

}  // namespace tree
