#include <sawIntuitiveResearchKit/mtsNeuralForceEstimation.h>

#include <iostream>
#include <sstream>
#include <vector>

std::string print_shape(const std::vector<std::int64_t>& v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

mtsNeuralForceEstimation::mtsNeuralForceEstimation() : is_loaded(false) {}

bool mtsNeuralForceEstimation::Load(const std::basic_string<ORTCHAR_T> & modelFile)
{
    environment = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "mtsNeuralForceEstimation");
    session = std::make_unique<Ort::Session>(environment, modelFile.c_str(), Ort::SessionOptions());

    Ort::AllocatorWithDefaultOptions allocator;
    
    for (std::size_t i = 0; i < session->GetInputCount(); i++) {
        input_names.emplace_back(session->GetInputNameAllocated(i, allocator).get());
        input_shapes = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    for (auto& s : input_shapes) {
        s = (s >= 0) ? s : 1;
    }

    for (std::size_t i = 0; i < session->GetOutputCount(); i++) {
        output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
    }

    is_loaded = true;
    return true;
}

bool mtsNeuralForceEstimation::Ready() const {
    return is_loaded;
}

vctDoubleVec mtsNeuralForceEstimation::infer_jf(const vctDoubleVec& jp, const vctDoubleVec& jv)
{
    CMN_ASSERT(is_loaded);

    std::vector<float> input_tensor_values(12);
    input_tensor_values[0] = jp[0];
    input_tensor_values[1] = jp[1];
    input_tensor_values[2] = jp[2];
    input_tensor_values[3] = jp[3];
    input_tensor_values[4] = jp[4];
    input_tensor_values[5] = jp[5];
    input_tensor_values[6] = jv[0];
    input_tensor_values[7] = jv[1];
    input_tensor_values[8] = jv[2];
    input_tensor_values[9] = jv[3];
    input_tensor_values[10] = jv[4];
    input_tensor_values[11] = jv[5];

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shapes));

    std::vector<const char*> input_names_char(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                    [&](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                    [&](const std::string& str) { return str.c_str(); });

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                        input_names_char.size(), output_names_char.data(), output_names_char.size());

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

        float* output = output_tensors[0].GetTensorMutableData<float>();
        vctDoubleVec output_vec(3);
        output_vec[0] = output[0];
        output_vec[1] = output[1];
        output_vec[2] = output[2];
        return output_vec;
    } catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }
}
