/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  (C) Copyright 2023 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _mtsNeuralForceEstimation_h
#define _mtsNeuralForceEstimation_h

#include <memory>
#include <iostream>
#include <string>

#include <cisstVector/vctDynamicMatrixTypes.h>
#include <cisstVector/vctDynamicVectorTypes.h>
#include <cisstVector/vctFixedSizeVectorTypes.h>

#include <../onnxruntime/include/onnxruntime_cxx_api.h>

class shape_t {
public:
    shape_t(std::vector<std::int64_t> value);

    std::vector<std::int64_t> value;

    std::size_t ndim() const;
    std::int64_t size() const;
    std::int64_t operator[](size_t) const;

    bool operator==(const shape_t& shape) const;
    bool operator!=(const shape_t& shape) const;

    void replace_unknowns();
    std::int64_t* data();
    const std::int64_t* data() const;
};

std::ostream& operator<<(std::ostream& s, const shape_t& shape);

class mtsNeuralForceEstimation {
private:
    bool is_loaded;

    Ort::Env environment;
    std::unique_ptr<Ort::Session> session;

    std::vector<shape_t> input_shapes;
    std::vector<std::string> input_names;
    std::vector<const char*> input_names_char;

    std::vector<shape_t> output_shapes;
    std::vector<std::string> output_names;
    std::vector<const char*> output_names_char;

    std::vector<Ort::Value> internal_states;

    bool Validate();
    void InitializeState();

public:
    mtsNeuralForceEstimation();
    ~mtsNeuralForceEstimation(){};

    bool Load(const std::basic_string<ORTCHAR_T> & modelFile);

    bool Ready() const;
    
    vct3 infer_jf(const vctDynamicVector<double>& jp, const vctDynamicVector<double>& jv);
};

#endif
