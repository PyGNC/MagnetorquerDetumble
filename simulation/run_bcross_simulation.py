import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "SatellitePlayground.jl"))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import GNCTestClient  # noqa
import magnetorquer_detumble  # noqa

# set up initial state
client = GNCTestClient.GNCTestClient()
client.register_state("control", [0.0, 0.0, 0.0])
client.register_state("ω", [0.1, 0.2, 0.3])
client.register_state("b", [0.1, 1.1, -0.2])

# TODO - determine these from the simulator
semi_major_axis = 1e4
inclination = np.deg2rad(97.6)
minimum_inertia_moment = magnetorquer_detumble.bcross.Controller.compute_minimum_inertia_moment(0.01*np.eye(3))
maximum_dipoles = 0.001*np.ones(3)
output_range = maximum_dipoles

# set up detumble controller
controller = magnetorquer_detumble.bcross.Controller(
    semi_major_axis,
    inclination,
    minimum_inertia_moment,
    maximum_dipoles,
    output_range
)

# simulate
