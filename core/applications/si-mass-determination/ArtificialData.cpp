/*
  Author(s):  Brendan Burkhart
  Created on: 2025-01-21

  (C) Copyright 2025 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cmath>
#include <iostream>
#include <fstream>
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

#include "MassDetermination.hpp"

namespace mass_determination {

std::vector<Sample> artificial_data(const robManipulator& m, double mounting_angle) {
    std::vector<Sample> samples;
    vctDoubleVec zero_velocity = vctDoubleVec(m.links.size(), 0.0);

    std::vector<double> yaw_angles = { -0.3 * cmnPI, -0.1 * cmnPI, 0.0, 0.2 * cmnPI, 0.4 * cmnPI, 0.5 * cmnPI };
    std::vector<double> pitch_angles = { -0.35 * cmnPI, -0.2 * cmnPI, -0.1 * cmnPI, 0.05 * cmnPI, 0.15 * cmnPI, 0.25 * cmnPI };
    std::vector<double> insertions = { 0.120, 0.160, 0.200 };

    vct3 gravity = vct3(0.0, std::sin(mounting_angle) * 9.81, std::cos(mounting_angle) * 9.81);

    for (const double yaw : yaw_angles) {
        for (const double pitch : pitch_angles) {
            for (const double insertion : insertions) {
                joint_pose reduced_joints = vctVec(3, yaw, pitch, insertion);
                joint_pose joints = reduced_to_full(yaw, pitch, insertion);
                joint_efforts efforts = m.CCG_MDH(joints, zero_velocity, gravity);
                joint_efforts measured = measured_efforts(efforts);

                samples.push_back({ reduced_joints, measured });
            }
        }
    }

    return samples;
}

}

int main(int argc, char * argv[])
{
    cmnCommandLineOptions options;
    std::string kinematic_config;
    double mounting_angle = 0.0;
    std::string output_name;

    options.AddOptionOneValue("c", "config",
                              "kinematic configuration file",
                              cmnCommandLineOptions::REQUIRED_OPTION, &kinematic_config);
    options.AddOptionOneValue("a", "angle",
                              "mounting angle of robot, zero is horizontal, positive is down",
                              cmnCommandLineOptions::OPTIONAL_OPTION, &mounting_angle);
    options.AddOptionOneValue("o", "output",
                              "output file name",
                              cmnCommandLineOptions::REQUIRED_OPTION, &output_name);
    if (!options.Parse(argc, argv, std::cerr)) {
        return -1;
    }

    robManipulator manipulator;
    if (!cmnPath::Exists(kinematic_config)) {
        std::cout << "ERROR: could not find provided kinematic configuration file" << std::endl;
        return -1;
    }

    std::cout << "Loading kinematic configuration from '" << kinematic_config << "'" << std::endl;
    robManipulator::Errno err = manipulator.LoadRobot(kinematic_config);
    if (err == robManipulator::Errno::ESUCCESS) {
        std::cout << "    Successfully loaded robot with " << manipulator.links.size() << " links" << std::endl;
    } else {
        std::cout << "    Failed to load robot: " << manipulator.mLastError << std::endl;
        return -1;
    }

    std::cout << "Creating artificial data samples..." << std::endl;
    std::vector<mass_determination::Sample> samples = mass_determination::artificial_data(manipulator, mounting_angle);

    std::cout << "Created " << samples.size() << " samples" << std::endl;
    
    std::fstream output{output_name, output.trunc | output.out};
    if (!output.is_open()) {
        std::cout << "Failed to open output file!" << std::endl;
    } else {
        for (auto& sample : samples) {
            output << sample << "\n";
        }
    }

    std::cout << "Artificial samples saved to '" << output_name << "'" << std::endl;
}
