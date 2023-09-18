#include <sawIntuitiveResearchKit/mtsContactDetection.h>

#include <iostream>
#include <sstream>
#include <vector>

template <typename T>
Ort::Value create_tensor(std::vector<T>& data, const shape_t& shape)
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(memory_info, data.data(), data.size(), shape.data(), shape.ndim());
    return tensor;
}

template <typename T>
Ort::Value zero_tensor(const shape_t& shape)
{
    std::vector<T> zeros(shape.size(), 0.0);
    return create_tensor(zeros, shape);
}

bool mtsContactDetection::Validate()
{
    return true;
}

void mtsContactDetection::InitializeState()
{
    internal_states.clear();

    for (size_t i = 1; i < input_shapes.size(); i++) {
        Ort::Value initial_state = zero_tensor<float>(input_shapes[i]);
        internal_states.push_back(std::move(initial_state));
    }
}

mtsContactDetection::mtsContactDetection() : is_loaded(false) {}

bool mtsContactDetection::Load(const std::basic_string<ORTCHAR_T> & modelFile)
{
    environment = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "mtsContactDetection");
    session = std::make_unique<Ort::Session>(environment, modelFile.c_str(), Ort::SessionOptions());

    Ort::AllocatorWithDefaultOptions allocator;

    for (std::size_t i = 0; i < session->GetInputCount(); i++) {
        auto name = session->GetInputNameAllocated(i, allocator);
        input_names.emplace_back(name.get());

        shape_t input_shape = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        input_shape.replace_unknowns();
        input_shapes.push_back(input_shape);
    }

    for (std::size_t i = 0; i < session->GetOutputCount(); i++) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(name.get());

        shape_t output_shape = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        output_shape.replace_unknowns();
        output_shapes.push_back(output_shape);
    }

    for (size_t i = 0; i < session->GetInputCount(); i++) {
        input_names_char.push_back(input_names.at(i).c_str());
    }

    for (size_t i = 0; i < session->GetOutputCount(); i++) {
        output_names_char.push_back(output_names.at(i).c_str());
    }

    if (!Validate()) {
        return false;
    }

    InitializeState();

    is_loaded = true;
    return true;
}

bool mtsContactDetection::Ready() const
{
    return is_loaded;
}

bool mtsContactDetection::infer(const vctDynamicVector<double>& jv, const vctDynamicVector<double>& jf)
{
    CMN_ASSERT(is_loaded);

    size_t n = input_shapes[0].size() / 2;
    std::vector<float> input_tensor_values(2 * n);
    for (size_t i = 0; i < n; i++) {
        input_tensor_values[i] = jv[i];
        input_tensor_values[n + i] = jf[i];
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(create_tensor<float>(input_tensor_values, input_shapes[0]));

    for (Ort::Value& state : internal_states) {
        input_tensors.push_back(std::move(state));
    }

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                        input_names_char.size(), output_names_char.data(), output_names_char.size());

        if (output_tensors.size() != output_shapes.size()) {
            throw Ort::Exception("received wrong number of outputs", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
        }

        float* output_values = output_tensors[0].GetTensorMutableData<float>();
        
        for (size_t i = 1; i < output_tensors.size(); i++) {
            internal_states[i - 1] = std::move(output_tensors[i]);
        }

        bool in_contact = output_values[0] > 0.5;

        return in_contact;
    } catch (const Ort::Exception& exception) {
        std::cout << "Error: running network inference: " << exception.what() << std::endl;
        exit(-1);
    }
}
