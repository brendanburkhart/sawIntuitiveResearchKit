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

#ifndef _mtsContactDetection_h
#define _mtsContactDetection_h

#include <memory>
#include <iostream>
#include <string>

#include <cisstVector/vctDynamicMatrixTypes.h>
#include <cisstVector/vctDynamicVectorTypes.h>
#include <cisstVector/vctFixedSizeVectorTypes.h>

#include <sawIntuitiveResearchKit/mtsNeuralForceEstimation.h>

class mtsContactDetection {
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
    mtsContactDetection();
    ~mtsContactDetection(){};

    bool Load(const std::basic_string<ORTCHAR_T> & modelFile);

    bool Ready() const;
    
    bool infer(const vctDynamicVector<double>& jv, const vctDynamicVector<double>& jf);
};

#endif
