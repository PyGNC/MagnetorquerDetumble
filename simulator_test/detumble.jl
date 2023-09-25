using LinearAlgebra
using SatelliteDynamics
using Plots
using SatellitePlayground
using PyCall

SP = SatellitePlayground

semi_major_axis = 400e3 + SatelliteDynamics.R_EARTH
inclination = deg2rad(50)
x_osc_0 = [semi_major_axis, 0.0, inclination, deg2rad(-1.0), 0.0, 0.0] # a, e, i, Ω, ω, M
q0 = [1.0, 0.0, 0.0, 0.0]
ω0 = 0.1 * [0.3, 0.1, -0.2]
ω0 = ω0 / norm(ω0) * deg2rad(50.0)
x0 = SP.state_from_osc(x_osc_0, q0, ω0)

control_limit_multiplier = 10.0
py4_dipole_limits = [0.06997731147540984,
    0.053130000000000004,
    0.06976756111111111] * control_limit_multiplier

py"""
import sys
sys.path.insert(0, "..")
"""
magnetorquer_detumble = pyimport("magnetorquer_detumble")
Controller = magnetorquer_detumble.Controller
Detumble = Controller(
    semi_major_axis,
    inclination,
    Controller._compute_minimum_inertia_moment([0.3 0 0; 0 0.3 0; 0 0 0.3]),
    py4_dipole_limits,
    py4_dipole_limits,
    true
)

function control_law(measurement)
    (state, env) = measurement

    m = Detumble.get_control(state.angular_velocity, env.b)
    return SP.Control(
        clamp.(m, -py4_dipole_limits, py4_dipole_limits)
    )
end


down_sample_rate = 100

function log_state(state)
    return [state.angular_velocity; norm(state.angular_velocity)]
end

function log_init(state)
    return [log_state(state)]
end

function log_step(hist, state)
    push!(hist, log_state(state))
end

function slow_rotation(state, env, i)
    return norm(state.angular_velocity) < deg2rad(0.1)
end

day = 60 * 60 * 24
time_step = 0.1
@time (data, time) = SP.simulate(control_law, max_iterations=day / time_step, dt=time_step,
    log_init=log_init, log_step=log_step, initial_condition=x0, terminal_condition=slow_rotation)
# 465.717581 seconds (1.48 G allocations: 94.596 GiB, 8.23% gc time, 0.02% compilation time: 82% of which was recompilation)

data = SP.vec_to_mat(data[1:down_sample_rate:end])

time = time[1:down_sample_rate:end]
time = time[1:size(data)[1]]
time /= 60

data = rad2deg.(data)

display(plot(time, data, title="DeTumbling", xlabel="Time (minutes)", ylabel="Angular Velocity (deg/s)", labels=["ω1" "ω2" "ω3" "ω"]))
