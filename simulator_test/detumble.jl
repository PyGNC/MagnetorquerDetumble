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

py4_dipole_limits = [0.06997731147540984,
    0.053130000000000004,
    0.06976756111111111]

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
    py4_dipole_limits
)

δ=0.053130000000000004*0.01

function saturate(x, r)
    if abs(x) <= δ
        return r
    elseif x > δ
        return r
    else # x < -δ
        return -r
    end
end

function control_law(measurement, t)
    (state, params) = measurement

    ᵇQⁿ = SP.quaternionToMatrix(state.attitude)'


    m = Detumble.get_control(state.angular_velocity, ᵇQⁿ * params.b)
    for i in 1:3
        m[i] = saturate(m[i], py4_dipole_limits[i])
    end

    @assert clamp.(m, -py4_dipole_limits, py4_dipole_limits) ≈ m
    
    return SP.Control(
        m
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

function log_end(hist)
    return SP.default_log_end(hist[1:down_sample_rate:end])
end

function slow_rotation(state, params, t, i)
    return norm(state.angular_velocity) < deg2rad(0.1)
end

day = 60 * 60 
time_step = 0.1
@time (data, time) = SP.simulate(control_law, max_iterations=day / time_step, dt=time_step,
    log_init=log_init, log_step=log_step, log_end=log_end, initial_condition=x0, terminal_condition=slow_rotation)

time = time[1:down_sample_rate:end]
time = time[1:size(data)[1]]
time /= 60

data = rad2deg.(data)

display(plot(time, data, title="DeTumbling with (δ = $δ)", xlabel="Time (minutes)", ylabel="Angular Velocity (deg/s)", labels=["ω1" "ω2" "ω3" "ω"]))
