import mujoco
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
from src.mpc_model import MPC
import matplotlib.pyplot as plt

if __name__ == "__main__":
    panda = load_robot_description("panda_mj_description")
    data = mujoco.MjData(panda)

    mpc = MPC(data=data, trajectory_id=0)

    # Create MPC model
    # mpc_model = create_model(data)
    # mpc = create_mpc(mpc_model)

    # sim = simulate_mpc_mujoco(mpc, panda, data)

    # Plot results
    joint_states = mpc.simulate()
    t = np.arange(len(joint_states[1])) * panda.opt.timestep

    plt.figure()
    for i in range(7):
        plt.plot(t, joint_states[i + 1], label=f"Joint {i+1}")
    # plt.plot(t, get_trajectory(), label="trajectory")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.legend()
    plt.show()
