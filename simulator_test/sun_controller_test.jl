#sun test

#structure similar to detumble.jl but adding in sun sensor measurement 
# to obtain the new control control_law

using Pkg
#Pkg.activate(@__DIR__)
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
#change maybe? 

day = 60 * 60 * 12
# day = 10
time_step = 0.1

py"""
import sys
sys.path.insert(0, "..")
"""
magnetorquer_detumble = pyimport("magnetorquer_detumble")

R_sat_imu = [
    0.0 1.0 0.0
    -1.0 0.0 0.0
    0.0 0.0 -1.0
]
mag_bias = 40e-6 * (2 * rand(3) .- 1.0) # bias in +/- 40uT
println("mag_bias_imu = $mag_bias")
begin
    PracticalController = magnetorquer_detumble.PracticalController
    ω_in = zeros(3)
    b_in = zeros(3)
    lux_in = zeros(6)

    PracticalDetumble = PracticalController(
        model.control_limit,
        model.control_limit,
        b_in,
        ω_in,
        lux_in,
        10 * pi,
        #added in
        which_controller = 1,
    )
    printed = false
    function practical_detumble_control(ω, b, lux, dt)
        global printed
        b_in .= b .+ mag_bias # magnetic field in imu frame
        ω_in .= ω # angular rate in imu frame
        lux_in .= lux #lux
        if !PracticalDetumble.mag_bias_estimate_complete
            PracticalDetumble.update_bias_estimate(dt)
            m = 0.0 * zeros(3)
        else
            if !printed && PracticalDetumble.mag_bias_estimate_complete
                println("\nestimated mag_bias_imu = $(PracticalDetumble.mag_bias)")
                printed = true
            end
            PracticalDetumble.new_mag = true
            m = PracticalDetumble.get_control(dt)
        end
        return m
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

function get_lux_measurement(state, env)

    #define parameters from models.jl
    standard_deviation = 0.01
    solar_lux =(1361 * 98)
    earth_albedo_lux = 0.4 * (1361 * 98)
    dark_condition_lux = 0.0

    #0.01 is the standard deviation of the sun sensor
    #defined in models.jl in pygnc repo
    sun_std_dev_matrix = I(3) .* 0.01

    sun_vector_eci = SatelliteDynamics.sun_position(env.time)
    #unix_time_s = epoch_to_unix_time(env.time)

    ᵇQⁿ = SP.quaternionToMatrix(state.attitude)'

    # since the sun vector is normalized and in the body frame,
    # each component is the cos() of the angle between the sun and that satellite face
    sun_vector_body = ᵇQⁿ * normalize(state.position .- sun_vector_eci) + sun_std_dev_matrix * randn(3)
    earth_vector_body = ᵇQⁿ * normalize(-state.position) + sun_std_dev_matrix * randn(3)
    # combine sun and earth albedo lux
    lux_vector_body = (solar_lux * sun_vector_body +
                        earth_albedo_lux * earth_vector_body)
    dark_measurement = dark_condition_lux * rand(3)
    sun_sensors_positive_faces = max.(dark_measurement, lux_vector_body)
    dark_measurement = dark_condition_lux * rand(3)
    sun_sensors_negative_faces = -min.(dark_measurement, lux_vector_body)
    sun_sensors = [sun_sensors_positive_faces; sun_sensors_negative_faces]

    return sun_sensors

end

function control_law(measurement)
    (state, env) = measurement

    dt = time_step

    #get the lux measurement
    lux_measurement = get_lux_measurement(state, env)

    m = practical_detumble_control(state.angular_velocity, env.b, lux_measurement, dt)
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

#@time 
(data, time) = SP.simulate(control_law, max_iterations=day / time_step, dt=time_step, environment=env,
    log_init=log_init, log_step=log_step, initial_condition=x0, terminal_condition=slow_rotation, model=model, silent=false)

# 465.717581 seconds (1.48 G allocations: 94.596 GiB, 8.23% gc time, 0.02% compilation time: 82% of which was recompilation)

# Old result: ω=0.129 at 3 hours
# New result: ω=0.156 at 3 hours

down_sample_rate = 10

data = data[1:down_sample_rate:end]
time = time[1:down_sample_rate:end]
time /= 60

data = rad2deg.(data)
detumble_plot = plot(time, data, title="DeTumbling", xlabel="Time (minutes)", ylabel="Angular Velocity (deg/s)", labels=["ω"])

#save the plot
savefig(detumble_plot, "sun_controller.png")
