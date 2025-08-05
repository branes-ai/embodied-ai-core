# Modern PyTorch Deployment Guide for Embodied AI

## Overview

The updated programs now use PyTorch's modern deployment stack instead of the deprecated TorchScript. Here's how to deploy these models to C++/Rust production systems.

## Deployment Methods

### 1. **torch.export() + ExecuTorch** (Recommended for Embedded)
- **Best for**: Edge devices, mobile, embedded systems
- **Format**: `.pt2` files
- **Benefits**: Optimized for inference, smaller memory footprint
- **Requirements**: PyTorch 2.1+, ExecuTorch runtime

### 2. **LibTorch C++ API** (Recommended for Servers)
- **Best for**: Server deployments, development flexibility
- **Format**: `.pth` weight files + model definition
- **Benefits**: Full PyTorch C++ API access
- **Requirements**: LibTorch C++ library

### 3. **ONNX Runtime** (Cross-platform)
- **Best for**: Cross-platform deployment
- **Format**: `.onnx` files
- **Benefits**: Language agnostic, hardware optimized
- **Requirements**: ONNX Runtime

## Updated Model Export Process

### Navigation Model
```python
# Export for ExecuTorch deployment
exported_program = torch.export.export(model, (example_input,))
torch.export.save(exported_program, 'navigation_model.pt2')

# Input: [batch_size, env_size² × 4] - Multi-channel environment state  
# Output: [batch_size, 4] - Q-values for 4 movement actions
```

### Manipulation Model  
```python
# Export vision-guided manipulation model
exported_program = torch.export.export(model, (example_image, example_state))
torch.export.save(exported_program, 'manipulation_model.pt2')

# Image input: [batch_size, 3, 400, 400] - RGB camera feed
# State input: [batch_size, 10] - Robot joint/gripper state
# Output: [batch_size, 4] - Action commands [dx, dy, dz, gripper]
```

### HRI Model
```python
# Export language understanding components separately
action_classifier_export = torch.export.export(action_classifier, (embedding,))
param_extractor_export = torch.export.export(param_extractor, (combined_input,))

# Command embedding: [batch_size, 768] - DistilBERT output
# Action output: [batch_size, 6] - Action type probabilities  
# Parameters: [batch_size, 10] - Action parameter vector
```

## C++ Deployment Examples

### Using ExecuTorch (Embedded)
```cpp
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/platform/runtime.h>

// Load exported model
auto method = torch::executor::util::LoadMethod("navigation_model.pt2");

// Prepare input tensor
std::vector<float> input_data(env_size * env_size * 4);
// ... populate input_data with environment state ...

torch::executor::EValue input_tensor = 
    torch::executor::util::CreateTensor(input_data, {1, input_size});

// Execute inference
auto outputs = method->execute({input_tensor});
auto q_values = outputs[0].toTensor();

// Extract action
int best_action = torch::argmax(q_values, 1).item<int>();
```

### Using LibTorch (Server)
```cpp
#include <torch/torch.h>
#include <torch/script.h>

// Recreate model architecture in C++
class DQNAgent : public torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    
public:
    DQNAgent(int state_size, int action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(state_size, 256));
        fc2 = register_module("fc2", torch::nn::Linear(256, 256));
        fc3 = register_module("fc3", torch::nn::Linear(256, 256));
        fc4 = register_module("fc4", torch::nn::Linear(256, action_size));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.2, is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::dropout(x, 0.2, is_training());
        x = torch::relu(fc3->forward(x));
        return fc4->forward(x);
    }
};

// Load trained weights
auto model = std::make_shared<DQNAgent>(state_size, 4);
torch::load(model, "navigation_model_weights.pth");
model->eval();

// Inference
torch::Tensor input = torch::randn({1, state_size});
torch::Tensor output = model->forward(input);
```

## Rust Deployment Examples

### Using tch-rs + ExecuTorch
```rust
use tch::{nn, Device, Tensor, Kind};

// Load ExecuTorch model (when Rust bindings available)
// For now, use tch-rs with weight loading

struct DQNAgent {
    vs: nn::VarStore,
    fc1: nn::Linear,
    fc2: nn::Linear, 
    fc3: nn::Linear,
    fc4: nn::Linear,
}

impl DQNAgent {
    fn new(vs: &nn::Path, state_size: i64) -> Self {
        Self {
            vs: vs.clone(),
            fc1: nn::linear(vs / "fc1", state_size, 256, Default::default()),
            fc2: nn::linear(vs / "fc2", 256, 256, Default::default()),
            fc3: nn::linear(vs / "fc3", 256, 256, Default::default()),
            fc4: nn::linear(vs / "fc4", 256, 4, Default::default()),
        }
    }
    
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1)
          .relu()
          .dropout(0.2, false) 
          .apply(&self.fc2)
          .relu()
          .dropout(0.2, false)
          .apply(&self.fc3)
          .relu()
          .apply(&self.fc4)
    }
}

// Load model
let vs = nn::VarStore::new(Device::Cpu);
let model = DQNAgent::new(&vs.root(), state_size);
vs.load("navigation_model_weights.pth")?;

// Inference
let input = Tensor::randn(&[1, state_size], (Kind::Float, Device::Cpu));
let output = model.forward(&input);
let action = output.argmax(1, false);
```

## Hardware Integration Patterns

### Real-time Control Loop
```cpp
class EmbodiedAIController {
    std::unique_ptr<NavigationModel> nav_model_;
    std::unique_ptr<ManipulationModel> manip_model_;
    std::unique_ptr<HRIModel> hri_model_;
    
public:
    void control_loop() {
        while (running_) {
            // Get sensor data
            auto camera_image = sensor_manager_.get_camera_frame();
            auto robot_state = sensor_manager_.get_robot_state();
            auto lidar_scan = sensor_manager_.get_lidar_scan();
            
            // Navigation inference
            auto nav_action = nav_model_->infer(lidar_scan, goal_position_);
            
            // Manipulation inference  
            auto manip_action = manip_model_->infer(camera_image, robot_state);
            
            // Execute actions
            robot_controller_.execute_navigation(nav_action);
            robot_controller_.execute_manipulation(manip_action);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20Hz
        }
    }
};
```

## Performance Considerations

### Memory Optimization
- **ExecuTorch**: ~10-50% smaller memory footprint than TorchScript
- **Quantization**: INT8 quantization for 4x memory reduction
- **Pruning**: Remove unused model parameters

### Latency Optimization  
- **Batch size 1**: Optimize for single-sample inference
- **SIMD**: Leverage vectorized operations
- **GPU**: Use CUDA/OpenCL for parallel computation

### Safety Features
- **Confidence thresholding**: Reject low-confidence predictions
- **Fallback behaviors**: Default actions for model failures
- **Bounds checking**: Validate all model outputs

## Migration from TorchScript

| TorchScript (Deprecated) | Modern Approach |
|-------------------------|-----------------|
| `torch.jit.trace()` | `torch.export.export()` |
| `torch.jit.script()` | Manual model definition |
| `.pt` files | `.pt2` files (ExecuTorch) |
| `torch::jit::load()` | ExecuTorch runtime |
| Limited optimization | Full graph optimization |

## Production Checklist

- [ ] **Model validation**: Test exported models match Python accuracy
- [ ] **Input validation**: Bounds checking and normalization  
- [ ] **Error handling**: Graceful degradation on model failures
- [ ] **Performance profiling**: Measure inference latency and memory usage
- [ ] **Safety verification**: Test edge cases and failure modes
- [ ] **Hardware integration**: Validate sensor data pipeline
- [ ] **Real-time constraints**: Verify control loop timing requirements

## Additional Resources

- [PyTorch Export Documentation](https://pytorch.org/docs/stable/export.html)
- [ExecuTorch Runtime](https://pytorch.org/executorch/)
- [LibTorch C++ API](https://pytorch.org/cppdocs/)
- [tch-rs Rust Bindings](https://github.com/LaurentMazare/tch-rs)

The modern deployment stack provides better performance, smaller binaries, and more flexible deployment options compared to the deprecated TorchScript approach.