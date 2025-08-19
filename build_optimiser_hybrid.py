#!/usr/bin/env python3
import opengen as og
import casadi.casadi as cs

#Problem setup
nu, nx = 2, 3                   # controls: [v, delta], states: [x, y, theta]
horizon_len = 20
wheelbase = 0.214
dt = 0.1
v_max, delta_max = 1.0, 0.35


def build_optimizer(name, v_limits, pos_weight, yaw_weight, yaw_final_weight, steer_weight):
    # Decision variables
    u = cs.SX.sym('u', nu * horizon_len)    # control inputs over horizon
    z0 = cs.SX.sym('z0', nx)                # start state
    ref = cs.SX.sym('ref', nx)               # goal state

    x, y, theta = z0[0], z0[1], z0[2]
    cost = 0
    prev_delta = 0

    # Loop over each timestep in horizon
    for t in range(0, nu * horizon_len, nu):
        v, delta = u[t], u[t + 1]

        # Bicycle model kinematics
        theta_dot = v * cs.tan(delta) / wheelbase
        x += dt * v * cs.cos(theta)
        y += dt * v * cs.sin(theta)
        theta += dt * theta_dot

        # Position tracking
        cost += pos_weight * ((x - ref[0]) ** 2 + (y - ref[1]) ** 2)

        # Gradually increase yaw importance along the horizon
        stage_yaw_weight = yaw_weight * (t / (nu * horizon_len))
        cost += stage_yaw_weight * (theta - ref[2]) ** 2

        # velocities and steering angles
        cost += 0.05 * (v ** 2) + steer_weight * (delta ** 2)

        # Penalize big changes in steering
        if t > 0:
            cost += 5.0 * ((delta - prev_delta) ** 2)
        prev_delta = delta

        # Penalize reversing
        if_penalty = cs.if_else(v < 0, 2.0 * (v ** 2), 0)
        cost += if_penalty

    # yaw alignment at final state
    cost += yaw_final_weight * (theta - ref[2]) ** 2

    # Input constraints
    umin = [v_limits[0], -delta_max] * horizon_len
    umax = [v_limits[1], delta_max] * horizon_len
    bounds = og.constraints.Rectangle(umin, umax)

    # Build the optimization problem
    problem = (
        og.builder.Problem(u, cs.vertcat(z0, ref), cost)
        .with_constraints(bounds)
    )

    # Build configuration
    build_cfg = (
        og.config.BuildConfiguration()
        .with_build_directory(f"my_optimizers/{name}")
        .with_build_mode("release")
        .with_build_python_bindings()
    )

    meta = og.config.OptimizerMeta().with_optimizer_name(name)
    solver_cfg = og.config.SolverConfiguration().with_tolerance(1e-5)

    # Build and compile
    builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_cfg, solver_cfg)
    builder.build()
    print(f"Optimizer built")


if __name__ == "__main__":
    build_optimizer(
        name="navigation_hybrid",
        v_limits=[-v_max, v_max],     # forward and reverse
        pos_weight=50,                # position tracking
        yaw_weight=20,                # yaw less important early on
        yaw_final_weight=500,         # precise yaw at goal
        steer_weight=1.0               # steering penalty
    )
