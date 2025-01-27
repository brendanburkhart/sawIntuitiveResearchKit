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

// Set all masses to 0.0 and all centers of mass to (0.0, 0.0, 0.0)
void clear_mass_data(robManipulator& m) {
    auto eye = vctFixedSizeMatrix<double,3,3>::Eye();

    for (auto& link : m.links) {
        link.MassData() = robMass(0.0, vct3(0.0), 0.0001 * eye, eye);
    }
}

// Set mass k to <mass>
void set_mass_k(robManipulator& m, size_t k, double mass) {
    clear_mass_data(m);

    auto eye = vctFixedSizeMatrix<double,3,3>::Eye();
    m.links.at(k).MassData() = robMass(mass, vct3(0.0, 0.0, 0.0), 0.0001 * eye, eye);
}

// Set mass k to <mass> and the kth center of mass to unit vector along <axis>
void set_com_k(robManipulator& m, size_t k, size_t axis, double mass) {
    clear_mass_data(m);

    auto eye = vctFixedSizeMatrix<double,3,3>::Eye();
    vct3 com(0.0, 0.0, 0.0);
    com[axis] = mass;

    m.links.at(k).MassData() = robMass(mass, com, 0.0001 * eye, eye);
}

Equations construct_equations(std::vector<Sample> samples, robManipulator& m, double mounting_angle) {
    vctDoubleVec zero_velocity = vctDoubleVec(m.links.size(), 0.0);

    size_t N = measured_efforts(zero_velocity).size();
    size_t n_dims = 4 * (m.links.size() - 1);

    vctDoubleMat equations(samples.size() * N, n_dims, VCT_COL_MAJOR);
    vctDoubleVec b(samples.size() * N);

    vct3 gravity = vct3(0.0, std::sin(mounting_angle) * 9.81, std::cos(mounting_angle) * 9.81);

    for (size_t sample_idx = 0; sample_idx < samples.size(); sample_idx++) {
        Sample& sample = samples[sample_idx];

        size_t idx = 0;
        for (size_t k = 0; k < m.links.size(); k++) {
            bool joint_mode = m.links[k].GetKinematics()->GetMode();
            if (joint_mode == robKinematics::Mode::PASSIVE) {
                continue;
            }

            joint_pose pose = reduced_to_full(sample.pose[0], sample.pose[1], sample.pose[2]);

            clear_mass_data(m);
            set_mass_k(m, k, 1.0);
            vctDoubleVec constant_efforts = measured_efforts(m.CCG_MDH(pose, zero_velocity, gravity));
            for (size_t p = 0; p < N; p++) {
                equations.at(sample_idx * N + p, 4 * idx) = constant_efforts[p];
            }

            for (size_t axis = 0; axis < 3; axis++) {
                clear_mass_data(m);
                set_com_k(m, k, axis, 1.0);

                vctDoubleVec efforts = measured_efforts(m.CCG_MDH(pose, zero_velocity, gravity));
                vctDoubleVec com_efforts = efforts - constant_efforts;
                for (size_t p = 0; p < N; p++) {
                    equations.at(sample_idx * N + p, 4 * idx + 1 + axis) = com_efforts[p];
                }
            }

            idx++;
        }

        for (size_t p = 0; p < N; p++) {
            b.at(sample_idx * N + p) = sample.efforts[p];
        }
    }

    return { equations, b };
}

}

int main(int argc, char * argv[])
{
    cmnCommandLineOptions options;
    std::string kinematic_config;
    std::string data_file;
    std::string output_file;
    double mounting_angle = 0.0;
    options.AddOptionOneValue("c", "config",
                              "kinematic configuration file",
                              cmnCommandLineOptions::REQUIRED_OPTION, &kinematic_config);
    options.AddOptionOneValue("a", "angle",
                              "mounting angle of robot, zero is horizontal, positive is down",
                              cmnCommandLineOptions::OPTIONAL_OPTION, &mounting_angle);
    options.AddOptionOneValue("i", "input",
                              "input data file name",
                              cmnCommandLineOptions::REQUIRED_OPTION, &data_file);
    options.AddOptionOneValue("o", "output",
                              "output file name",
                              cmnCommandLineOptions::REQUIRED_OPTION, &output_file);
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

    std::fstream sample_data{data_file, sample_data.in};
    if (!sample_data.is_open()) {
        std::cout << "Failed to open input file" << std::endl;
        return -1;
    }

    std::vector<mass_determination::Sample> samples = {};
    mass_determination::Sample sample;
    while (sample_data >> sample) {
        samples.push_back(sample);
    }

    std::cout << "Loaded " << samples.size() << " data samples" << std::endl;

    std::cout << "Constructing mass determination equations" << std::endl;
    auto equations = mass_determination::construct_equations(samples, manipulator, mounting_angle);

    std::fstream output{output_file, output.trunc | output.out};
    if (!output.is_open()) {
        std::cout << "Failed to open output file!" << std::endl;
    } else {
        output << equations << std::endl;
        std::cout << "Equations saved to '" << output_file << "'" << std::endl;
    }
}
