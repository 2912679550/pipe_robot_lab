# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import time
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

PIP_ROBOT_CONFIG = ArticulationCfg(
    # 文件路径
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/vulcan/Academic/pipe_robot_lab/source/model/pipe_robot/USD/pipe_robot_mini/pipe_robot_mini.usd",
        # 刚体属性（示例）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # Articulation 根/求解器属性（允许自碰撞等）
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0 , 0.0, 0.0),
        joint_pos={},
        joint_vel={}
    ),
    actuators={
        "all_joints_default": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=5.0,
            damping=0.5,
            # effort_limit_sim=.0,
        ),
    }
    
    
)

class SingleRobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/lightDistant", spawn=sim_utils.DistantLightCfg(intensity=5000.0))
    # 使用 replace，把 prim_path 设置为场景中的路径（支持多环境时用 {ENV_REGEX_NS}）
    PipeRobot = PIP_ROBOT_CONFIG.replace(prim_path="/World/pipe_robot")

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=5000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    
    # 以下代码适合加载静态物体模型
    # cfg = sim_utils.UsdFileCfg(
    #     usd_path="/home/vulcan/Academic/pipe_robot_lab/source/moudle/pipe_robot/urdf/pipe_robot/pipe_robot_changed.usd",
    # )
    # cfg.func("/World/pipe_robot", cfg, translation=(0.0, 0.0, 1.0))
    
    # 加载动态机器人模型 ArticulationCfg
    pipe_robot = PIP_ROBOT_CONFIG
    pipe_robot.spawn.func("/World/pipe_robot", pipe_robot.spawn)
    # 加载机器人（注意：这里不需要加translation，USD内部的根变换已包含位置）
    # cfg_robot.func("/World/pipe_robot", cfg_robot)


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01, 
        device=args_cli.device,
        # gravity=(0.0, 0.0, 0.0),        # 零重力
        gravity=(0.0, 0.0, -9.81),
        )
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    # design_scene()
    # 用 InteractiveScene 自动实例化 ArticulationCfg（包括 init_state）
    scene_cfg = SingleRobotSceneCfg(1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")


    # # Simulate physics
    prev_sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        # 检测 pause -> resume：若之前物理 dt 为 0（paused），现在恢复 (>0)，则尝试唤醒/重置 Articulation
        cur_sim_dt = sim.get_physics_dt()
        if prev_sim_dt == 0 and cur_sim_dt > 0:
            try:
                # InteractiveScene 会把配置属性作为属性名挂载到 scene 上（例如 PipeRobot）
                if hasattr(scene, "PipeRobot"):
                    art = getattr(scene, "PipeRobot")
                    # Articulation 对象通常有 reset() 方法用于清空内部缓冲并唤醒刚体
                    if hasattr(art, "reset"):
                        art.reset()
            except Exception:
                pass
        prev_sim_dt = cur_sim_dt



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    