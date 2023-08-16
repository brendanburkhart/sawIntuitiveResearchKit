#include <sawIntuitiveResearchKit/mtsNeuralForceEstimation.h>

#include <iostream>
#include <sstream>
#include <vector>

shape_t::shape_t(std::vector<std::int64_t> value) : value(value) {}

std::size_t shape_t::ndim() const
{
    return value.size();
}

std::int64_t shape_t::size() const
{
    int64_t size = 1;
    for (auto dim : value) {
        size = size * dim;
    }

    return size;
}

std::int64_t shape_t::operator[](size_t i) const
{
    return value.at(i);
}

bool shape_t::operator==(const shape_t& shape) const
{
    if (ndim() != shape.ndim()) {
        return false;
    }

    for (size_t i = 0; i < ndim(); i++) {
        if ((*this)[i] != shape[i]) {
            return false;
        }
    }

    return true;
}

bool shape_t::operator!=(const shape_t& shape) const
{
    return !(*this == shape);
}

void shape_t::replace_unknowns()
{
    for (auto& v : value) {
        v = (v >= 0) ? v : 1;
    }
}

std::int64_t* shape_t::data()
{
    return value.data();
}

const std::int64_t* shape_t::data() const
{
    return value.data();
}

std::ostream& operator<<(std::ostream& s, const shape_t& shape)
{
    for (size_t dim = 0; dim < shape.ndim() - 1; dim++) {
        s << shape[dim] << "x";
    }
    s << shape[shape.ndim() - 1];

    return s;
}

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

template <size_t input, size_t output>
bool mtsNeuralForceEstimation<input, output>::Validate()
{
    if (input_shapes.size() == 0 || input_shapes.size() != output_shapes.size()) {
        std::cout << "Expected network to have at least one input/output, and to have equal number of inputs and outputs."
                  << " Instead, network has " << input_shapes.size() << " input(s) and " << output_shapes.size() << " output(s).\n";
        return false;
    }

    if (input_shapes[0].ndim() != 3 || input_shapes[0].size() != 2 * input) {
        std::cout << "Network input shape mismatch: found " << input_shapes[0]
                  << ", but should have 3 dimensions and total size of " << 2 * input << std::endl;
        return false;
    }

    if (output_shapes[0].ndim() > 3 || output_shapes[0].size() != output) {
        std::cout << "Network output shape mismatch: found " << output_shapes[0]
                  << ", but should have at most 3 dimensions and total size of " << output << std::endl;
        return false;
    }

    for (size_t i = 1; i < input_shapes.size(); i++) {
        if (input_shapes[i] != output_shapes[i]) {
            std::cout << "Network input/output pairs for hidden state must have the same shape,"
                      << " but input/output pair " << i << " are " << input_shapes[i]
                      << " and " << output_shapes[i] << " respectively.\n";

            return false;
        }
    }

    return true;
}

template <size_t input, size_t output>
void mtsNeuralForceEstimation<input, output>::InitializeState()
{
    internal_states.clear();

    for (size_t i = 1; i < input_shapes.size(); i++) {
        Ort::Value initial_state = zero_tensor<float>(input_shapes[i]);
        internal_states.push_back(std::move(initial_state));
    }
}

template <size_t input, size_t output>
mtsNeuralForceEstimation<input, output>::mtsNeuralForceEstimation() : is_loaded(false) {}

template <size_t input, size_t output>
bool mtsNeuralForceEstimation<input, output>::Load(const std::basic_string<ORTCHAR_T> & modelFile)
{
    environment = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "mtsNeuralForceEstimation");
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

template <size_t input, size_t output>
bool mtsNeuralForceEstimation<input, output>::Ready() const
{
    return is_loaded;
}

template <size_t input, size_t output>
typename mtsNeuralForceEstimation<input, output>::output_t mtsNeuralForceEstimation<input, output>::infer_jf(const input_t& jp, const input_t& jv)
{
    CMN_ASSERT(is_loaded);

    std::vector<float> input_tensor_values(2 * input);
    for (size_t i = 0; i < input; i++) {
        input_tensor_values[i] = jp[i];
        input_tensor_values[input + i] = jv[i];
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
        output_t output_vec;
        for (size_t i = 0; i < output; i++) {
            output_vec[i] = output_values[i];
        }

        for (size_t i = 1; i < output_tensors.size(); i++) {
            internal_states[i - 1] = std::move(output_tensors[i]);
        }

        return output_vec;
    } catch (const Ort::Exception& exception) {
        std::cout << "Error: running network inference: " << exception.what() << std::endl;
        exit(-1);
    }
}

template class mtsNeuralForceEstimation<6, 3>;
template class mtsNeuralForceEstimation<3, 3>;
