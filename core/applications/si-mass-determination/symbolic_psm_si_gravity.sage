# Author(s):  Brendan Burkhart
# Created on: 2025-01-21
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

class Link:
    def __init__(self, name, is_revolute, mass, com, alpha, a, theta, d):
        self.name = name
        self.revolute = is_revolute

        self.mass = mass
        self.com = com

        self.alpha = alpha
        self.a = a
        self.theta = theta
        self.d = d

    """Position of next link frame (when q = 0)"""
    def next_position(self):
        return vector(SR, [ self.a, -self.d * sin(self.alpha), self.d * cos(self.alpha) ])
    
    """Orientation of next link frame given q"""
    def orientation(self, q):
        theta = self.theta + q if self.revolute else self.theta

        return Matrix(SR, 3, [[cos(theta), -sin(theta), 0],
                              [sin(theta)*cos(self.alpha), cos(theta)*cos(self.alpha), -sin(self.alpha)],
                              [sin(theta)*sin(self.alpha), cos(theta)*sin(self.alpha), cos(self.alpha)]
                             ])

# parallelogram link lengths and angles
var("l1x,l1y,l2x,l2y")
angle_1 = atan2(l1y, l1x)
angle_2 = atan2(l2y, l2x)
angle_3 = 0

links = [
    Link("yaw",        True,  var("m1"), vector(SR, [var("cx1"), var("cy1"), var("cz1")]), pi/2,  0,                 pi/2,              0),
    Link("inter_yaw",  True,  0,         vector(SR, [0, 0, 0]),                            -pi/2, -0.0712,           pi/2,              0),
    Link("pitch1",     True,  var("m3"), vector(SR, [var("cx3"), var("cy3"), var("cz3")]), 0,     0.2913,            angle_1,           0),
    Link("pitch2",     True,  var("m4"), vector(SR, [var("cx4"), var("cy4"), var("cz4")]), 0,     sqrt(l1x^2+l1y^2), angle_2 - angle_1, 0),
    Link("pitch3",     True,  var("m5"), vector(SR, [var("cx5"), var("cy5"), var("cz5")]), 0,     sqrt(l2x^2+l2y^2), -angle_2,          0),
    Link("insertion1", False, var("m6"), vector(SR, [var("cx6"), var("cy6"), var("cz6")]), -pi/2, 0.0602,            pi,                -0.1800),
    Link("insertion2", False, var("m7"), vector(SR, [var("cx7"), var("cy7"), var("cz7")]), 0,     -0.0320,           0,                 -0.1215)
]

qs = [ var("y"), var("p"), var("d") ]
g = var("g")

def rnea(qs, g):
    q = [qs[0], 0, qs[1], -qs[1], qs[1], qs[2]/2, qs[2]/2]

    acceleration = vector(SR, [0.0, 0.0, g]) # linear acceleration of link 0
    z0 = vector(SR, [0.0, 0.0, 1.0])
    forces = []

    for i, link in enumerate(links):
        A = link.orientation(q[i]).transpose() # rotation from link i to link i+1
        acceleration = A * acceleration # linear acceleration
        f = (link.mass * acceleration).simplify_full()
        forces.append(f)

    efforts = []
    f = vector(SR, [0, 0, 0])
    n = vector(SR, [0, 0, 0])

    for i in range(len(links) - 1, -1, -1):
        link = links[i]
        A = identity_matrix(SR, 3) if i+1 == len(links) else links[i+1].orientation(q[i+1])
        A = A.simplify_full()
        p = vector(SR, [0, 0, 0]) if i+1 == len(links) else links[i+1].next_position()
        p = p.simplify_full()
        s = links[i].com

        # force and moment exerted on i by i-1
        n = A * n + s.cross_product(forces[i]) + p.cross_product(A * f)
        n = n.simplify_full()
        f = A * f + forces[i]
        f = f.simplify_full()

        effort = n.dot_product(z0) if link.revolute else f.dot_product(z0)
        effort = effort.simplify_full()
        efforts = [effort] + efforts

    efforts = [ t.simplify_full() for t in efforts] 

    return efforts
