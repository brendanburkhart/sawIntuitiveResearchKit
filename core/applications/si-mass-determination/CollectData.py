#!/usr/bin/env python3

# Author(s):  Brendan Burkhart
# Created on: 2025-01-24
#
# (C) Copyright 2025 Johns Hopkins University (JHU), All Rights Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

import argparse
import sys
import crtk
import dvrk
import math
import numpy
import time

class DataCollection:
    def __init__(self, ral, arm_name, timeout, output_file):
        self.ral = ral
        self.arm = dvrk.arm(ral, arm_name, timeout)
        self.output_file = output_file

    def home(self):
        print("Homing arm")
        if not self.arm.enable(10) or not self.arm.home(10):
            sys.exit("    ! failed to enable home arm within timeout")

    # make sure insertion is past cannula to enable Cartesian commands
    def prepare_cartesian(self):
        name = self.arm.name()
        if (name.endswith('PSM1') or name.endswith('PSM2')
            or name.endswith('PSM3') or name.endswith('ECM')):
            goal = numpy.copy(self.arm.setpoint_jp()[0])

            goal[0] = 0.0
            goal[1] = 0.0
            goal[2] = 0.12
            goal[3] = 0.0
            self.arm.move_jp(goal).wait()

    def collect(self):
        yaw_angles = [ -0.45 * math.pi, -0.3 * math.pi, -0.1 * math.pi, 0.0, 0.2 * math.pi, 0.4 * math.pi, 0.5 * math.pi ]
        pitch_angles = [ -0.35 * math.pi, -0.2 * math.pi, -0.1 * math.pi, 0.05 * math.pi, 0.15 * math.pi, 0.25 * math.pi ]
        insertions = [ 0.120, 0.160, 0.200 ]

        goal = numpy.copy(self.arm.setpoint_jp()[0])
        goal.fill(0.0)

        samples = []
        total_samples = len(yaw_angles) * len(pitch_angles) * len(insertions)

        for i, yaw in enumerate(yaw_angles):
            pitches = pitch_angles if i % 2 == 0 else reversed(pitch_angles)
            for j, pitch in enumerate(pitches):
                inserts = insertions if j % 2 == 0 else reversed(insertions)
                for insertion in inserts:
                    goal[0:3] = [yaw, pitch, insertion]
                    self.arm.move_jp(goal).wait()
                    time.sleep(1.0)
                    poses = []
                    efforts = []
                    for _ in range(10):
                        p, v, e, t = self.arm.measured_js()
                        poses.append(p[0:3])
                        efforts.append(e[0:3])
                        time.sleep(0.1)

                    pose = numpy.mean(poses, axis=0)
                    efforts = numpy.mean(efforts, axis=0)
                    samples.append((pose, efforts))
                    print(f"\r{int(100 * len(samples)/total_samples)}% done", end='', flush=True)

        print()
        self.prepare_cartesian()

        return samples

    def run(self):
        self.ral.check_connections()
        self.home()
        self.prepare_cartesian()
        samples = self.collect()
        print(f"Collected {len(samples)} samples")

        with open(self.output_file, 'w') as f:
            for sample in samples:
                pose_string = " ".join([str(d) for d in sample[0]])
                effort_string = " ".join([str(d) for d in sample[1]])
                f.write(pose_string + " | " + effort_string + "\n")

        print(f"Collected data saved to {self.output_file}")

if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True,
                        choices=['PSM1', 'PSM2', 'PSM3'],
                        help = 'arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help = 'output file name')
    args = parser.parse_args(argv)

    ral = crtk.ral('CollectData')
    timeout = 0.1 # seconds
    application = DataCollection(ral, args.arm, timeout, args.output)
    ral.spin_and_execute(application.run)
