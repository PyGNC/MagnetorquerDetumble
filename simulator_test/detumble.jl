using LinearAlgebra
using SatelliteDynamics
using Plots
using SatellitePlayground
using PyCall

SP = SatellitePlayground

begin 
    # Initial Conditions
    semi_major_axis = 400e3 + SatelliteDynamics.R_EARTH
    inclination = deg2rad(50)
    x_osc_0 = [semi_major_axis, 0.0, inclination, deg2rad(-1.0), 0.0, 0.0] # a, e, i, Ω, ω, M
    q0 = [1.0, 0.0, 0.0, 0.0]
    ω0 = 0.1 * [0.3, 0.1, -0.2]
    ω0 = ω0 / norm(ω0) * deg2rad(10.0)
    x0 = SP.state_from_osc(x_osc_0, q0, ω0)
end

begin
    # Model
    model = copy(SP.pqmini_model)
    control_limit_multiplier = 1.0
    model.control_limit *= control_limit_multiplier

    env = copy(SP.default_environment)
    env.config = SP.EnvironmentConfig(
        n_gravity=10,
        m_gravity=10,
        include_drag=true,
        include_solar_radiation_pressure=true,
        include_sun_gravity=true,
        include_moon_gravity=true,
        include_gravity_gradient_torque=true)

end

# day = 60 * 60 * 24
day = 60 * 60 * 3
# day = 10
time_step = 0.1

py"""
import sys
sys.path.insert(0, "..")
"""
magnetorquer_detumble = pyimport("magnetorquer_detumble")

begin
    PracticalController= magnetorquer_detumble.PracticalController
    PracticalDetumble = PracticalController(
        model.control_limit,
        model.control_limit,
        sense_time=1.0,
        actuate_time=1.0,
    )
    function practical_detumble_control(ω, b, dt)
        return PracticalDetumble.get_control(ω, b, dt)
    end
end
begin
    Controller = magnetorquer_detumble.Controller
    Detumble = Controller(
        semi_major_axis,
        inclination,
        Controller._compute_minimum_inertia_moment(model.inertia),
        model.control_limit,
        model.control_limit,
    )
    function detumble_control(ω, b, dt)
        return Detumble.get_control(ω, b)
    end
end

function control_law(measurement)
    (state, env) = measurement

    dt = time_step

    m = practical_detumble_control(state.angular_velocity, env.b, dt)
    return SP.Control(
        m
    )
end

function log_state(state)
    return norm(state.angular_velocity)
end

function log_init(state)
    return []
end

function log_step(hist, state)
    push!(hist, log_state(state))
end

function slow_rotation(state, env, i)
    return norm(state.angular_velocity) < deg2rad(0.1)
end

@time (data, time) = SP.simulate(control_law, max_iterations=day / time_step, dt=time_step, environment=env,
    log_init=log_init, log_step=log_step, initial_condition=x0, terminal_condition=slow_rotation, model=model)
# 465.717581 seconds (1.48 G allocations: 94.596 GiB, 8.23% gc time, 0.02% compilation time: 82% of which was recompilation)

# Old result: ω=0.129 at 3 hours
# New result: ω=0.156 at 3 hours

down_sample_rate = 10

data = data[1:down_sample_rate:end]
time = time[1:down_sample_rate:end]
time /= 60

data = rad2deg.(data)

display(plot(time, data, title="DeTumbling", xlabel="Time (minutes)", ylabel="Angular Velocity (deg/s)", labels=["ω"]))
