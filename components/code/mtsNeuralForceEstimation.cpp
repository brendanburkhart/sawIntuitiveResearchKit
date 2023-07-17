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

mtsNeuralForceEstimation::mtsNeuralForceEstimation(const std::basic_string<ORTCHAR_T> & modelFile)
    : environment(ORT_LOGGING_LEVEL_WARNING, "mtsNeuralForceEstimation"),
      session(environment, modelFile.c_str(), Ort::SessionOptions())
{
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::int64_t> input_shapes;
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }
    // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
    for (auto& s : input_shapes) {
        if (s < 0) {
        s = 1;
        }
    }

    // print name/shape of outputs
    std::vector<std::string> output_names;
    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
    }
}
