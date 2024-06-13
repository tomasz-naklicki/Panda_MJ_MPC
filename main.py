import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description
import time
import numpy as np
import do_mpc

panda = load_robot_description("panda_mj_description")
data = mujoco.MjData(panda)


with mujoco.viewer.launch_passive(panda, data) as viewer:
    start = time.time()
    while viewer.is_running():  # and time.time() - start < 10:
        step_start = time.time()
        mujoco.mj_step(panda, data)
        q_current = np.array(data.qpos).reshape(-1, 1)
        q_dot_current = np.array(data.qvel).reshape(-1, 1)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()

        time_until_next_step = panda.opt.timestep - (time.time() - step_start)
        print(f"JOINT STATES: {q_current}")
        print(f"VELOCITIES: {q_dot_current}")
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
