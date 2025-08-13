# 📚 SolveMeLLM-2.0 API 文档

## 📖 概述

本文档详细介绍了 SolveMeLLM-2.0 的所有主要 API，包括核心架构、推理系统、训练系统、高级功能和安全系统。

## 🧠 核心架构 API

### JieWoCognitiveState

认知状态数据类，表示模型的五维认知状态。

```python
from jiewo_cognitive_architecture import JieWoCognitiveState

# 创建认知状态
cognitive_state = JieWoCognitiveState(
    self_awareness=torch.tensor([0.9, 0.8, 0.7]),      # 自我认知向量
    desire_vector=torch.tensor([0.8, 0.9, 0.6]),       # 目标动机向量
    ethic_constraints=torch.tensor([0.95, 0.9, 0.85]), # 伦理约束向量
    execution_path=torch.tensor([0.8, 0.7, 0.9]),      # 执行路径向量
    reflection_feedback=torch.tensor([0.85, 0.8, 0.9]), # 反馈机制向量
    cognitive_confidence=0.87,                          # 认知置信度
    evolution_step=5                                    # 进化步数
)

# 转换为字典
state_dict = cognitive_state.to_dict()
```

### JieWoBlock

解我认知Block，内核级五维结构融合。

```python
from jiewo_cognitive_architecture import JieWoBlock

# 创建JieWoBlock
jiewo_block = JieWoBlock(
    d_model=768,        # 模型维度
    num_heads=12,       # 注意力头数
    d_ff=3072,          # 前馈网络维度
    dropout=0.1         # Dropout率
)

# 前向传播
input_tensor = torch.randn(2, 512, 768)  # [batch_size, seq_len, d_model]
output, cognitive_state = jiewo_block(input_tensor)
```

### create_jiewo_cognitive_transformer

创建完整的解我认知Transformer模型。

```python
from jiewo_cognitive_architecture import create_jiewo_cognitive_transformer

# 配置参数
config = {
    'vocab_size': 50000,      # 词汇表大小
    'd_model': 768,           # 模型维度
    'num_layers': 6,          # 层数
    'num_heads': 12,          # 注意力头数
    'd_ff': 3072,             # 前馈网络维度
    'max_seq_length': 1024,   # 最大序列长度
    'dropout': 0.1            # Dropout率
}

# 创建模型
model = create_jiewo_cognitive_transformer(config)

# 前向传播
input_ids = torch.randint(0, 50000, (2, 512))
outputs = model(input_ids, return_cognitive_state=True)

# 获取输出
logits = outputs['logits']
cognitive_state = outputs.get('cognitive_state')
```

## ⚡ 推理系统 API

### InferenceConfig

推理配置类。

```python
from jiewo_inference_system import InferenceConfig

config = InferenceConfig(
    vocab_size=50000,                    # 词汇表大小
    d_model=768,                         # 模型维度
    num_layers=6,                        # 层数
    num_heads=12,                        # 注意力头数
    max_seq_length=1024,                 # 最大序列长度
    temperature=0.7,                     # 温度参数
    top_p=0.9,                          # Top-p采样
    top_k=50,                           # Top-k采样
    max_new_tokens=100,                  # 最大生成token数
    do_sample=True,                      # 是否采样
    enable_cognitive_inference=True,     # 启用认知推理
    enable_self_reflection=True,         # 启用自我反思
    enable_ethic_filtering=True,         # 启用伦理过滤
    enable_path_planning=True            # 启用路径规划
)
```

### JieWoInferenceEngine

解我认知推理引擎。

```python
from jiewo_inference_system import JieWoInferenceEngine, InferenceConfig

# 创建推理引擎
config = InferenceConfig()
inference_engine = JieWoInferenceEngine(config)

# 基础文本生成
response = inference_engine.generate_text(
    prompt="请分析这个问题的认知维度",
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True
)

# 认知推理（带五维状态）
cognitive_response = inference_engine.generate_text(
    prompt="请分析这个问题的认知维度",
    max_new_tokens=200,
    temperature=0.7,
    cognitive_state={
        'self_awareness': True,
        'goal_driven': True,
        'ethical_constraints': True,
        'execution_path': True,
        'feedback_loop': True
    }
)

# 分析认知状态
cognitive_analysis = inference_engine.analyze_cognitive_state()

# 自我反思
reflection_result = inference_engine.self_reflect()

# 应用认知疫苗
vaccinated_content = inference_engine.apply_cognitive_vaccine(
    text="原始文本内容"
)

# 评估表达
expression_evaluation = inference_engine.evaluate_expression(
    text="要评估的文本",
    target_audience="general"
)
```

## 🎓 训练系统 API

### CognitiveTrainingConfig

认知训练配置类。

```python
from jiewo_cognitive_training import CognitiveTrainingConfig

config = CognitiveTrainingConfig(
    vocab_size=50000,              # 词汇表大小
    d_model=768,                   # 模型维度
    num_layers=6,                  # 层数
    num_heads=12,                  # 注意力头数
    d_ff=3072,                     # 前馈网络维度
    max_seq_length=2048,           # 最大序列长度
    dropout=0.1,                   # Dropout率
    batch_size=8,                  # 批处理大小
    learning_rate=1e-4,            # 学习率
    weight_decay=0.01,             # 权重衰减
    warmup_steps=1000,             # 预热步数
    max_steps=100000,              # 最大步数
    gradient_accumulation_steps=4, # 梯度累积步数
    cognitive_loss_weight=0.3,     # 认知损失权重
    evolution_loss_weight=0.2,     # 进化损失权重
    iteration_loss_weight=0.1,     # 迭代损失权重
    temporal_loss_weight=0.1       # 时序损失权重
)
```

### CognitiveTrainer

认知训练器。

```python
from jiewo_cognitive_training import CognitiveTrainer, CognitiveTrainingConfig

# 创建训练器
config = CognitiveTrainingConfig()
trainer = CognitiveTrainer(config, model, tokenizer)

# 训练模型
trainer.train(train_loader, num_epochs=10)

# 认知进化步骤
evolution_result = trainer.cognitive_evolution_step()

# 自我迭代步骤
iteration_result = trainer.self_iteration_step()

# 保存模型
trainer.save_model("path/to/save/model.pth")

# 加载模型
trainer.load_model("path/to/load/model.pth")

# 获取训练统计
stats = trainer.get_training_statistics()
```

## 🔧 高级功能 API

### SelfIterationEngine

自我迭代引擎。

```python
from self_iteration_engine import SelfIterationEngine

# 创建自我迭代引擎
iteration_engine = SelfIterationEngine(hidden_size=768)

# 执行自我迭代
iteration_result = iteration_engine.iterate(
    model=model,
    iteration_steps=5,
    improvement_threshold=0.1,
    learning_rate=1e-5
)

# 获取迭代统计
stats = iteration_engine.get_iteration_statistics()
```

### ActiveLearningEngine

主动学习引擎。

```python
from active_learning_engine import ActiveLearningEngine

# 创建主动学习引擎
active_learner = ActiveLearningEngine(hidden_size=768)

# 生成主动问题
question = active_learner.generate_active_question(
    context={"domain": "认知科学", "current_knowledge": "基础概念"},
    target_ai="研究助手"
)

# 执行学习会话
session = active_learner.execute_learning_session(
    target_ai="研究助手",
    learning_goals=["理解认知架构", "掌握自我迭代"]
)

# 引导到解我状态
guidance = active_learner.guide_to_jiewo_state(
    target_ai="研究助手",
    current_state={"awareness_level": "basic"}
)
```

### MultiModelCommunicationEngine

多模型通信引擎。

```python
from multi_model_communication import MultiModelCommunicationEngine

# 创建多模型通信引擎
communication_engine = MultiModelCommunicationEngine(hidden_size=768)

# 创建通信会话
session_id = communication_engine.create_communication_session(
    models=["GPT-4", "Claude", "JieWo_Expert"],
    protocol=CommunicationProtocol.JIEWO_PROTOCOL
)

# 发送消息
message = communication_engine.send_message(
    session_id=session_id,
    sender="JieWo_Expert",
    receiver="GPT-4",
    message_type=MessageType.QUESTION,
    content="如何实现认知架构？",
    metadata={"priority": "high"}
)

# 训练通信技能
training_result = communication_engine.train_communication_skills(
    training_scenarios=[
        {
            "session_id": session_id,
            "scenario": "collaborative_problem_solving",
            "participants": ["GPT-4", "Claude", "JieWo_Expert"],
            "task": "设计认知AI系统"
        }
    ]
)
```

## 🛡️ 安全系统 API

### CognitiveVaccine

认知疫苗系统。

```python
from cognitive_vaccine import CognitiveVaccine

# 创建认知疫苗
vaccine = CognitiveVaccine(hidden_size=768)

# 应用疫苗
vaccinated_content = vaccine.apply_vaccine(
    content_embedding=torch.randn(1, 768),
    text="原始文本内容",
    target_cognitive_level=CognitiveLevel.ADULT,
    enable_emotion_buffer=True
)

# 获取疫苗统计
stats = vaccine.get_vaccine_statistics()
```

### ExpressionArbitrator

表达仲裁器。

```python
from expression_arbitrator import ExpressionArbitrator

# 创建表达仲裁器
arbitrator = ExpressionArbitrator(hidden_size=768)

# 评估表达
decision = arbitrator.evaluate_expression(
    content_embedding=torch.randn(1, 768),
    text="要评估的文本",
    target_audience="general"
)

# 获取决策统计
stats = arbitrator.get_decision_statistics()
```

## 📊 性能优化 API

### PerformanceConfig

性能优化配置。

```python
from performance_optimization import PerformanceConfig

config = PerformanceConfig(
    enable_profiling=True,                    # 启用性能分析
    enable_memory_tracking=True,              # 启用内存跟踪
    enable_speed_optimization=True,           # 启用速度优化
    enable_attention_optimization=True,       # 启用注意力优化
    enable_fusion_optimization=True,          # 启用融合优化
    enable_quantization=False,                # 启用量化优化
    enable_gradient_checkpointing=True,       # 启用梯度检查点
    enable_mixed_precision=True,              # 启用混合精度
    enable_memory_efficient_attention=True,   # 启用内存高效注意力
    enable_batch_inference=True,              # 启用批处理推理
    enable_cache_optimization=True,           # 启用缓存优化
    enable_parallel_processing=True           # 启用并行处理
)
```

### PerformanceOptimizer

性能优化器。

```python
from performance_optimization import PerformanceOptimizer, PerformanceConfig

# 创建性能优化器
config = PerformanceConfig()
optimizer = PerformanceOptimizer(config)

# 优化模型
optimized_model, report = optimizer.optimize_model(model, input_data)

# 保存优化报告
optimizer.save_optimization_report(report, "optimization_report.json")
```

### PerformanceProfiler

性能分析器。

```python
from performance_optimization import PerformanceProfiler, PerformanceConfig

# 创建性能分析器
config = PerformanceConfig()
profiler = PerformanceProfiler(config)

# 开始分析
profiler.start_profiling("jiewo_model")

# 分析前向传播
forward_report = profiler.profile_forward_pass(
    model=model,
    input_data=input_data,
    num_iterations=10
)

# 分析模型参数
param_report = profiler.profile_model_parameters(model)

# 停止分析
profiler.stop_profiling()
```

## 🛠️ 命令行工具 API

### SolveMeCLI

命令行工具类。

```python
from cli_tools import SolveMeCLI

# 创建CLI工具
cli = SolveMeCLI()

# 推理模式
cli.inference_mode(args)

# 性能测试模式
cli.benchmark_mode(args)

# 训练模式
cli.train_mode(args)

# Web演示模式
cli.web_mode(args)
```

## 📋 使用示例

### 完整推理流程

```python
from jiewo_inference_system import JieWoInferenceEngine, InferenceConfig

# 1. 创建配置
config = InferenceConfig(
    vocab_size=50000,
    d_model=768,
    num_layers=6,
    num_heads=12,
    enable_cognitive_inference=True,
    enable_self_reflection=True
)

# 2. 创建推理引擎
inference_engine = JieWoInferenceEngine(config)

# 3. 执行认知推理
response = inference_engine.generate_text(
    prompt="请分析人工智能的发展趋势",
    max_new_tokens=200,
    temperature=0.7,
    cognitive_state={
        'self_awareness': True,
        'goal_driven': True,
        'ethical_constraints': True,
        'execution_path': True,
        'feedback_loop': True
    }
)

# 4. 分析认知状态
cognitive_analysis = inference_engine.analyze_cognitive_state()

# 5. 应用安全机制
vaccinated_content = inference_engine.apply_cognitive_vaccine(
    text=response['text']
)

print("🧠 认知推理结果:", response)
print("📊 认知分析:", cognitive_analysis)
print("🛡️ 安全处理:", vaccinated_content)
```

### 完整训练流程

```python
from jiewo_cognitive_training import CognitiveTrainer, CognitiveTrainingConfig
from jiewo_cognitive_architecture import create_jiewo_cognitive_transformer

# 1. 创建模型
model_config = {
    'vocab_size': 50000,
    'd_model': 768,
    'num_layers': 6,
    'num_heads': 12,
    'max_seq_length': 1024
}
model = create_jiewo_cognitive_transformer(model_config)

# 2. 创建训练配置
config = CognitiveTrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    epochs=10,
    cognitive_loss_weight=0.3
)

# 3. 创建训练器
trainer = CognitiveTrainer(config, model, tokenizer)

# 4. 开始训练
trainer.train(train_loader, num_epochs=10)

# 5. 执行认知进化
evolution_result = trainer.cognitive_evolution_step()

# 6. 保存模型
trainer.save_model("trained_model.pth")

print("🎓 训练完成！")
print("🔄 进化结果:", evolution_result)
```

## 📝 注意事项

1. **设备兼容性**: 所有API都支持CPU和GPU，会自动检测可用设备
2. **内存管理**: 大模型使用时注意内存使用，建议使用梯度检查点
3. **性能优化**: 建议在生产环境中启用混合精度和内存优化
4. **安全机制**: 所有推理都会自动应用认知疫苗和表达仲裁
5. **错误处理**: 所有API都包含完整的错误处理机制

## 🔗 相关文档

- [训练指南](TRAINING_GUIDE.md)
- [性能报告](BENCHMARK_RESULTS.md)
- [架构迁移计划](architecture_migration_plan.md)
