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

        self.goal = numpy.zeros_like(self.arm.setpoint_jp()[0])

    def _init_progress_indicator(self, total_samples):
        self.start_time = time.time()
        self.total_samples = total_samples
        self.samples = 0
        self.last_indicator_length = 0

    def _update_progress_indicator(self, samples_collected):
        progress = int(100 * samples_collected/self.total_samples)
        elapsed_s = max(1.0, time.time() - self.start_time)
        speed = samples_collected / elapsed_s
        remaining_s = int((self.total_samples - samples_collected) / speed)
        remaining_time = f"{remaining_s} seconds" if remaining_s < 120 else f"{int(remaining_s/60)} minutes"

        if samples_collected == self.total_samples:
            message = "100% done!"
            end = '\n'
        else:
            message = f"{progress}% done, estimated time remaining: {remaining_time}"
            end = ''

        stick_out = max(0, self.last_indicator_length - len(message))
        self.last_indicator_length = len(message)
        padding = " " * stick_out

        print(f"\r{message}{padding}", end=end, flush=True)

    """Given three equal-length list A, B, and C this creates an iterator returning all possible
       element combinations (a,b,c), with only one index of the tuple changing at a time"""
    def _zig_zag(self, a_list, b_list, c_list):
        for i, a in enumerate(a_list):
            for j, b in enumerate(b_list):
                for k, c in enumerate(c_list):
                    yield (a, b, c)
                c_list = list(reversed(c_list))
            b_list = list(reversed(b_list))

    def _sample(self, yaw, pitch, insertion, reps):
        self.goal[0:3] = [yaw, pitch, insertion]
        self.arm.move_jp(self.goal).wait()
        time.sleep(0.50)
        poses = []
        efforts = []
        for _ in range(reps):
            p, v, e, t = self.arm.measured_js()
            poses.append(p[0:3])
            efforts.append(e[0:3])
            time.sleep(0.05)

        pose = numpy.mean(poses, axis=0)
        efforts = numpy.mean(efforts, axis=0)
        return pose, efforts

    def collect(self):
        short_yaw_angles = [ a * math.pi for a in [ -0.45, -0.3, 0.0, 0.2, 0.5 ]]
        yaw_angles = [ a * math.pi for a in [ -0.5, -0.3, -0.1, 0.15, 0.35, 0.45 ]]
        pitch_angles = [a * math.pi for a in [ -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3 ]]
        # pitch_angles = [a * math.pi for a in [ -0.2, -0.1, 0.0, 0.1, 0.2 ]]
        insertions = [ 0.050, 0.125, 0.200 ]

        # hack to avoid tracking error when moving to initial position
        # possibly due to disabling disturbance observers? but doesn't seem to happen later
        self.goal[0:3] = [-0.3 * math.pi, -0.3 * math.pi, 0.1]
        self.arm.move_jp(self.goal).wait()

        samples = []
        total_samples = (len(yaw_angles) + len(short_yaw_angles)) * len(pitch_angles) * len(insertions)
        self._init_progress_indicator(total_samples)

        # zig-zag along yaw first
        for yaw, pitch, insertion in self._zig_zag(short_yaw_angles, pitch_angles, insertions):
            pose, efforts = self._sample(yaw, pitch, insertion, 10)
            samples.append((pose, efforts))
            self._update_progress_indicator(len(samples))

        # then along pitch first
        for pitch, yaw, insertion in self._zig_zag(pitch_angles, yaw_angles, insertions):
            pose, efforts = self._sample(yaw, pitch, insertion, 10)
            samples.append((pose, efforts))
            self._update_progress_indicator(len(samples))

        return samples

    def run(self):
        self.ral.check_connections()
        self.home()
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
                        choices=['PSM1', 'PSM2', 'PSM3', 'ECM'],
                        help = 'arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help = 'output file name')
    args = parser.parse_args(argv)

    ral = crtk.ral('CollectData')
    timeout = 0.1 # seconds
    application = DataCollection(ral, args.arm, timeout, args.output)
    ral.spin_and_execute(application.run)
