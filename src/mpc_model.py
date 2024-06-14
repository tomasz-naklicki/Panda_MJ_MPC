import numpy as np
import casadi as ca
import do_mpc
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description
import time
import matplotlib.pyplot as plt


class MPC:

    def __init__(self, data, trajectory_id: int):
        self.data = data
        self.trajectory_id = trajectory_id
        mujoco.mj_forward(data.model, data)
        self.model = self.create_model()
        self.mpc = self.create_mpc(self.model)

    def get_trajectory(self, trajectory_id: int, t_now):
        if trajectory_id == 0:
            traj = np.sin(np.linspace(0, 2 * np.pi, 21)) * np.ones((7, 21))
            t_now_scaled = t_now / 2
            k = int(t_now_scaled % 21)
            return traj.T[k]
            # return np.array([0, np.pi / 2, np.pi / 3, 0, 0.5, 0, 1])

    def get_inertia_matrix(self):
        nv = self.data.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.data.model, M, self.data.qM)
        return M[:7, :7]

    def get_coriolis(self):
        return self.data.qfrc_bias

    def get_gravity_forces(self):
        return self.data.qfrc_gravcomp

    def create_model(self):
        model_type = "continuous"
        model = do_mpc.model.Model(model_type)

        # Define the states (joint positions and velocities)
        q = model.set_variable(var_type="_x", var_name="q", shape=(7, 1))
        q_dot = model.set_variable(var_type="_x", var_name="q_dot", shape=(7, 1))

        # Define the time variable trajectory
        target_joint_states = model.set_variable(
            var_type="_tvp", var_name="target_joint_states", shape=(7, 1)
        )

        # Define the control inputs (joint torques)
        tau = model.set_variable(var_type="_u", var_name="tau", shape=(7, 1))

        mujoco.mj_forward(self.data.model, self.data)  # initialize values

        M = self.get_inertia_matrix()
        C = self.get_coriolis()[:7].reshape(1, 7)
        G = self.get_gravity_forces()[:7]

        q_ddot = ca.mtimes(ca.inv(M), (tau - ca.mtimes(C, q_dot) - G))

        model.set_rhs("q", q_dot)
        model.set_rhs("q_dot", q_ddot)

        model.setup()
        return model

    def create_mpc(self, model):
        mpc = do_mpc.controller.MPC(model)
        n_horizon = 5
        t_step = 0.05

        setup_mpc = {
            "n_horizon": n_horizon,
            "t_step": 0.05,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 3,
            "collocation_ni": 2,
            "store_full_solution": True,
        }
        mpc.set_param(**setup_mpc)
        # trajectory = self.get_trajectory(self.trajectory_id)

        mterm = ca.sumsqr(
            model.x["q"] - model.tvp["target_joint_states"]
        )  # Terminal cost
        lterm = mterm  # Stage cost

        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {t_now}")
            traj = self.get_trajectory(trajectory_id=self.trajectory_id, t_now=t_now)
            for k in range(n_horizon + 1):
                tvp_template["_tvp", k, "target_joint_states"] = traj
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(tau=1e-2)  # Regularization term for control inputs

        # Define constraints
        mpc.bounds["lower", "_x", "q"] = -np.pi
        mpc.bounds["upper", "_x", "q"] = np.pi

        mpc.bounds["lower", "_u", "tau"] = -10
        mpc.bounds["upper", "_u", "tau"] = 10

        mpc.setup()
        return mpc

    def _simulate_mpc_mujoco(self, mpc, panda, data):
        x0 = np.zeros((14, 1))
        mpc.x0 = x0
        mpc.set_initial_guess()

        joint_states = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

        with mujoco.viewer.launch_passive(panda, data) as viewer:
            start = time.time()
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(panda, data)

                q_current = np.array(data.qpos).reshape(-1, 1)
                q_dot_current = np.array(data.qvel).reshape(-1, 1)
                x0[:7] = q_current[:7]
                x0[7:] = q_dot_current[:7]

                for i in range(7):
                    joint_states[i + 1].append(q_current[i])

                u0 = mpc.make_step(x0)
                data.ctrl[:7] = u0.flatten()

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        data.time % 2
                    )

                viewer.sync()
                time_until_next_step = panda.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        return joint_states

    def simulate(self):
        return self._simulate_mpc_mujoco(self.mpc, self.data.model, self.data)
