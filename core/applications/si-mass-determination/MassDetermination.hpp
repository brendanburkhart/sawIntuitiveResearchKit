/*
  Author(s):  Brendan Burkhart
  Created on: 2025-01-24

  (C) Copyright 2025 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef MassDetermination_hpp
#define MassDetermination_hpp

#include <iostream>
#include <sstream>
#include <vector>

// cisst/saw
#include <cisstCommon/cmnCommandLineOptions.h>
#include <cisstCommon/cmnPath.h>
#include <cisstNumerical/nmrLSqLin.h>
#include <cisstRobot/robManipulator.h>
#include <cisstVector/vctDynamicMatrixTypes.h>
#include <cisstVector/vctDynamicVectorTypes.h>
#include <cisstVector/vctFixedSizeMatrixTypes.h>
#include <cisstVector/vctFixedSizeVectorTypes.h>
#include <cisstVector/vctTransformationTypes.h>

namespace mass_determination {

using joint_efforts = vctVec;
using joint_pose = vctVec;

class Sample {
public:
    joint_pose pose;
    joint_efforts efforts;
};

class Equations {
public:
    vctMat A;
    vctVec b;
};

// Convert reduced joint positions used by virtual PSM Si DH params to equivalent
// joint positions for the physical DH parameters
joint_pose reduced_to_full(double yaw, double pitch, double insertion) {
    joint_pose joint_positions(7, 0.0);

    joint_positions[0] = yaw;             
    joint_positions[1] = 0.0;             // passive/virtual joint
    joint_positions[2] = pitch;           // parallelogram joint 1
    joint_positions[3] = -pitch;          // parallelogram joint 2
    joint_positions[4] = pitch;           // parallelogram joint 3
    joint_positions[5] = 0.5 * insertion; // insertion stage 1
    joint_positions[6] = 0.5 * insertion; // insertion stage 2 

    return joint_positions;
}

// given efforts for all joints (both active and passive),
// returns the efforts actually measured by the PSM Si
joint_efforts measured_efforts(const joint_efforts& all_efforts) 
{
    double yaw_effort = all_efforts[0];
    // joint 1 is passive
    // physical joints 2-4 are all connected to one motor, joint 3 is oriented opposite
    double pitch_effort = all_efforts[2] - all_efforts[3] + all_efforts[4];
    // joints 5, 6 are cascaded together so we only use joint 6
    double insertion_effort = all_efforts[6];

    return vctDoubleVec(3, yaw_effort, pitch_effort, insertion_effort);
}

std::ostream& operator<<(std::ostream& st, const Equations& equations)
{
    size_t n = equations.A.rows();
    if (equations.b.size() != n) {
        std::cout << "ERROR: equations data sizes are incompatible" << std::endl;
        return st;
    }

    for (size_t i = 0; i < n; i++) {
        st << equations.A.Row(i) << " " << equations.b[i] << "\n";
    }

    return st;
}

constexpr char sample_delimiter = '|';

std::ostream& operator<<(std::ostream& st, const Sample& sample)
{
    st << sample.pose << sample_delimiter << sample.efforts;
    return st;
}

vctVec read_vec(std::string data) {
    std::stringstream ss(data);
    double value;
    std::vector<double> values;

    while (ss >> value) {
        values.push_back(value);
    }
    
    vctVec output(values.size(), 0.0);
    for (size_t i = 0; i < values.size(); i++) {
        output[i] = values[i];
    }

    return output;
}

// No validation or robustness, only use on output of operator<< above
std::istream& operator>>(std::istream& st, Sample& sample)
{
    std::string line;
    std::getline(st, line);
    if (!st.good()) { return st; }

    std::stringstream line_ss(line);
    std::string pose, efforts;

    std::getline(line_ss, pose, sample_delimiter);
    std::getline(line_ss, efforts);
    
    sample.pose = read_vec(pose);
    sample.efforts = read_vec(efforts);

    return st;
}

}

#endif // MassDetermination_hpp
