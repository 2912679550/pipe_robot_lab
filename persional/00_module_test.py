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
import torch 
import math
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
        usd_path="/home/vulcan/Academic/pipe_robot_lab/source/model/pipe_robot/USD/pipe_robot_rename/pipe_robot_rename.usd",

        # 刚体属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # Articulation 根/求解器属性（允许自碰撞等）
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,  # 位置迭代次数
            solver_velocity_iteration_count=4,  # 速度迭代次数
        ),
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0 , 0.0, 0.0),
        joint_pos={
            r"main_steer_.*": 0.0,  # 所有主动轮舵向
            r"main_wheel_.*": 0.0,  # 所有主动轮轮速
            r"up_arm_.*": 0.0,      # 所有上臂关节(机身与大臂连接点)
            r"mid_arm_.*": 0.0,     # 所有中臂关节(大臂与小臂连接点)
            r"tail_arm_.*": 0.0,    # 所有尾臂关节(小臂与末端辅助轮连接点)
            r"assist_steer_.*": 0.0, # 所有辅助轮舵向
            r"assist_wheel_.*": 0.0, # 所有辅助轮轮速
            r"bend_.*": 0.0,         # 所有弯折变形关节
        },
        joint_vel={
            r"main_steer_.*": 0.0,  # 所有主动轮舵向
            r"main_wheel_.*": 1.0,  # 所有主动轮轮速
            r"up_arm_.*": 0.0,      # 所有上臂关节(机身与大臂连接点)
            r"mid_arm_.*": 0.0,     # 所有中臂关节(大臂与小臂连接点)
            r"tail_arm_.*": 0.0,    # 所有尾臂关节(小臂与末端辅助轮连接点)
            r"assist_steer_.*": 0.0, # 所有辅助轮舵向
            r"assist_wheel_.*": 1.0, # 所有辅助轮轮速
            r"bend_.*": 0.0,         # 所有弯折变形关节
        }
    ),
    actuators={
        # 配置所有舵轮的舵向
        "steer": ImplicitActuatorCfg(
            joint_names_expr=["main_steer_.*","assist_steer_.*"],
            effort_limit_sim={
                "main_steer_.*":    100.0,
                "assist_steer_.*":  100.0,
            },
            velocity_limit_sim=     2.0,
            stiffness=              5.0,
            damping=                0.5,
            # effort_limit_sim=.0,
        ),
        
        # 配置所有舵轮的轮子
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["main_wheel_.*","assist_wheel_.*"],
            effort_limit_sim={
                "main_wheel_.*":    20.0,
                "assist_wheel_.*":  20.0,
            },
            velocity_limit_sim=     20.0,
            stiffness=              0.0,
            damping=                1.0,
            # effort_limit_sim=50.0,
        ),
        
        # 配置变形夹持用arm关节
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["up_arm_.*","mid_arm_.*","tail_arm_.*"],
            effort_limit_sim={
                "up_arm_.*":    100.0,
                "mid_arm_.*":   200.0,
                "tail_arm_.*":  100.0,
            },
            velocity_limit_sim= {
                "up_arm_.*":    1.0,
                "mid_arm_.*":   1.0,
                "tail_arm_.*":  1.0,
            },
            stiffness=          500.0,
            damping=            5.0,
        ),
        # 配置机身弯折关节
        "bend": ImplicitActuatorCfg(
            joint_names_expr=["bend_.*"],
            effort_limit_sim=   200.0,
            velocity_limit_sim= 1.0,
            stiffness=          500.0,
            damping=            10.0,
        )
    }
    
    
)

class SingleRobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/lightDistant", spawn=sim_utils.DistantLightCfg(intensity=5000.0))
    
    # 新增：圆柱管道体
    pipe_obstacle = AssetBaseCfg(
        prim_path="/World/PipeObstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.18,        # 直径 360mm
            height=2.0,         # 长度 2m
            rigid_props=None,   # 无刚体属性 = 静态 (Static)
            collision_props=sim_utils.CollisionPropertiesCfg(), # 开启碰撞
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)), # 简单材质
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            # 默认圆柱沿Z轴，需绕X轴旋转90度使其沿Y轴: (w, x, y, z) = (cos(45), sin(45), 0, 0)
            rot=(0.70711, 0.70711, 0.0, 0.0),
        ),
    )

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
    
    # ---  获取机器人对象和关节索引 ---
    robot = scene["PipeRobot"]
    
    # 查找各类关节的索引
    # 注意：正则表达式中的 .* 已经被修正
    arm_indices, _ = robot.find_joints(".*arm_.*|bend_.*")     # arm 和 bend 关节
    steer_indices, _ = robot.find_joints(".*steer_.*")         # steer 关节
    wheel_indices, _ = robot.find_joints(".*wheel_.*")         # wheel 关节
    
    print(f"Arm/Bend indices: {arm_indices}")
    print(f"Steer indices: {steer_indices}")
    print(f"Wheel indices: {wheel_indices}")


    # 初始化控制器计时器
    sim_time = 0.0

    # # Simulate physics
    prev_sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        # --- 控制逻辑开始 ---
        if sim.is_playing():
            # 1. 准备位置目标张量 (基于默认位置初始化)
            # 维度: [num_envs, num_joints]
            pos_targets = robot.data.default_joint_pos.clone()
            
            # 2. Arm & Bend: -5° ~ 5° (正弦波控制位置)
            arm_amp = math.radians(5.0)
            arm_wave = arm_amp * math.sin(sim_time * 2.0)
            pos_targets[:, arm_indices] += arm_wave
            
            # 3. Steer: -10° ~ 10° (正弦波控制位置)
            steer_amp = math.radians(10.0)
            steer_wave = steer_amp * math.sin(sim_time * 1.5)
            pos_targets[:, steer_indices] += steer_wave
            
            # 应用位置目标 (这将驱动 stiffness > 0 的关节，即 arms, bend, steer)
            # 对于 wheel (stiffness=0)，这个位置目标会被忽略（这意味着不会产生回正力矩）
            # robot.set_joint_position_target(pos_targets)

            # 4. Wheel: 持续转动 (速度控制)
            # 准备速度目标张量
            vel_targets = torch.zeros_like(robot.data.default_joint_vel)
            
            # 设置轮子目标速度 (例如 5.0 rad/s)
            wheel_speed = 10.0 
            vel_targets[:, wheel_indices] = wheel_speed
            
            # 应用速度目标 (这将驱动 damping > 0 的关节)
            # 对于 wheel (stiffness=0, damping>0)，这将产生力矩追踪目标速度
            # 对于其他关节 (stiffness>0)，通常应设为 0 或者期望轨迹的导数。这里设为 0 作为阻尼项。
            robot.set_joint_velocity_target(vel_targets)

            # 更新时间
            sim_time += sim.get_physics_dt()

        # --- 控制逻辑结束 ---

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    