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

#include <cisstVector/vctDynamicMatrixTypes.h>
#include <cisstVector/vctDynamicVectorTypes.h>

#include <../onnxruntime/include/onnxruntime_cxx_api.h>
#include <string>

class mtsNeuralForceEstimation {
protected:
    Ort::Env environment;
    Ort::Session session;

    std::vector<std::int64_t> input_shapes;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

public:
    mtsNeuralForceEstimation(const std::basic_string<ORTCHAR_T> & modelFile);
    ~mtsNeuralForceEstimation(){};
    
    vctDoubleVec infer_jf(const vctDoubleVec& jp, const vctDoubleVec& jv);
};

#endif