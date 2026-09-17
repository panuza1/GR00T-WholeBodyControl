#pragma once

// CPU-only inference for the local MuJoCo build. Production keeps its TensorRT engines.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class OrtSimEngine {
public:
    bool Initialize(const std::string& model_path, bool use_fp16 = false) {
        if (use_fp16) {
            std::cerr << "Simulation ONNX backend only supports float32" << std::endl;
            return false;
        }
        try {
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetInterOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
            if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) return false;

            Ort::AllocatorWithDefaultOptions allocator;
            input_name_ = session_->GetInputNameAllocated(0, allocator).get();
            output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
            if (input_name_ != "obs_dict" ||
                (output_name_ != "action" && output_name_ != "encoded_tokens")) return false;

            auto input_type = session_->GetInputTypeInfo(0);
            auto output_type = session_->GetOutputTypeInfo(0);
            auto input_info = input_type.GetTensorTypeAndShapeInfo();
            auto output_info = output_type.GetTensorTypeAndShapeInfo();
            if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
            input_shape_ = input_info.GetShape();
            const auto output_shape = output_info.GetShape();
            if (input_shape_.size() != 2 || output_shape.size() != 2 ||
                input_shape_[0] != 1 || output_shape[0] != 1 ||
                input_shape_[1] <= 0 || output_shape[1] <= 0) return false;
            input_.assign(static_cast<size_t>(input_shape_[1]), 0.0f);
            output_.assign(static_cast<size_t>(output_shape[1]), 0.0f);
            if (output_name_ == "action" && output_.size() != 29) return false;
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "Simulation ONNX model load failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool CaptureGraph() const { return session_ != nullptr; } // No CUDA graph on CPU.
    bool IsInitialized() const { return session_ != nullptr && !input_.empty(); }
    size_t GetInputDimension() const { return input_.size(); }
    size_t GetTokenDimension() const { return output_.size(); }
    std::vector<float>& GetInputBuffer() { return input_; }
    std::vector<float>& GetActionBuffer() { return output_; }
    std::vector<float>& GetTokenBuffer() { return output_; }
    bool Infer() { return Run(); }
    bool Encode() { return Run(); }

private:
    bool Run() {
        if (!IsInitialized() || !std::all_of(input_.begin(), input_.end(),
                                             [](float x) { return std::isfinite(x); })) return false;
        try {
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto tensor = Ort::Value::CreateTensor<float>(memory, input_.data(), input_.size(),
                                                           input_shape_.data(), input_shape_.size());
            const char* input_name = input_name_.c_str();
            const char* output_name = output_name_.c_str();
            auto values = session_->Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1,
                                        &output_name, 1);
            const auto info = values[0].GetTensorTypeAndShapeInfo();
            if (info.GetElementCount() != output_.size()) return false;
            const float* result = values[0].GetTensorData<float>();
            if (!std::all_of(result, result + output_.size(),
                             [](float x) { return std::isfinite(x); })) return false;
            std::copy_n(result, output_.size(), output_.begin());
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "Simulation ONNX inference failed: " << e.what() << std::endl;
            return false;
        }
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "sonic-sim"};
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<float> input_;
    std::vector<float> output_;
};

using PolicyEngine = OrtSimEngine;
using EncoderEngine = OrtSimEngine;
