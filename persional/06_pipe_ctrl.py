# 2026.01.18: 经过初步测试后形成的完整Demo运行文件，期望配置管道检测机器人可按照按键控制夹持在管道表面并沿管道前进
import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Pipe Robot Control Demo Script.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import carb.input
import omni.appwindow # 新增导入
import math
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp


# =============================================================================
# 1. 机器人资产配置 (Robot Asset Configuration)
# =============================================================================
PIPE_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/vulcan/Academic/pipe_robot_lab/source/model/pipe_robot/USD/pipe_robot_rename/pipe_robot_rename.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            r"main_steer_.*": 0.0,  # 所有主动轮舵向
            r"main_wheel_.*": 0.0,  # 所有主动轮轮速
            r"up_arm_.*": 0.0,      # 所有上臂关节(机身与大臂连接点)
            r"mid_arm_.*": 0.8,     # 所有中臂关节(大臂与小臂连接点)
            r"tail_arm_.*": 0.0,    # 所有尾臂关节(小臂与末端辅助轮连接点)
            r"assist_steer_.*": 0.0, # 所有辅助轮舵向
            r"assist_wheel_.*": 0.0, # 所有辅助轮轮速
            r"bend_.*": 0.0,         # 所有弯折变形关节
        },
        joint_vel={
            r".*": 0.0,
        }
    ),
    actuators={
        "steer": ImplicitActuatorCfg(
            joint_names_expr=["main_steer_.*", "assist_steer_.*"],
            effort_limit_sim={
                "main_steer_.*":    100.0,
                "assist_steer_.*":  100.0,
            },
            velocity_limit_sim=     2.0,
            stiffness=              5.0,
            damping=                0.5,
            # effort_limit_sim=.0,
        ),
        "wheel": ImplicitActuatorCfg( # 轮子: 适合速度控制 (Stiffness=0)
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
        "bend": ImplicitActuatorCfg(
            joint_names_expr=["bend_.*"],
            effort_limit_sim=   200.0,
            velocity_limit_sim= 1.0,
            stiffness=          500.0,
            damping=            10.0,
        )
    }
)


# =============================================================================
# 2. 场景配置 (Scene Configuration)
# =============================================================================
@configclass
class PipeRobotSceneCfg(InteractiveSceneCfg):
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg()
    )
    # 光照
    light = AssetBaseCfg(
        prim_path="/World/lightDistant", 
        spawn=sim_utils.DistantLightCfg(intensity=5000.0)
    )
    # 管道障碍物 (静态)
    pipe_obstacle = AssetBaseCfg(
        prim_path="/World/PipeObstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.18,        # 直径 360mm
            height=2.0,         # 长度 2m
            rigid_props=None,   # 静态 (Static)
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5), # 中心位置
            rot=(0.70711, 0.70711, 0.0, 0.0), # 沿Y轴放置 (绕X轴转90度)
        ),
    )
    # 机器人 (必须使用 {ENV_REGEX_NS} 占位符)
    robot: ArticulationCfg = PIPE_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# =============================================================================
# 3. 动作配置 (Actions Configuration)
# =============================================================================
@configclass
class ActionsCfg:
    # 1. 轮速控制 (Velocity Control)
    # 注意：使用 Cfg 后缀的配置类
    wheels = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*wheel_.*"], 
        scale=1.0,
    )

    # 2. 机械臂位置控制 (Position Control)
    arms = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*arm_.*", "bend_.*"],
        scale=1.0,
        use_default_offset=True,
    )
    
    # 3. 舵向位置控制 (Position Control)
    steer = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*steer_.*"],
        scale=1.0,
        use_default_offset=True,
    )


# =============================================================================
# 4. 观测配置 (Observations Configuration)
# =============================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 简单观测：关节位置和速度
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
    policy = PolicyCfg()


# =============================================================================
# 5. 事件配置 (Events Configuration)
# =============================================================================
@configclass
class EventCfg:
    # Reset 时重置关节位置到默认附近 (微小随机)
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# =============================================================================
# 6. 其他 RL 必需配置 (Rewards, Terminations)
# =============================================================================
@configclass
class RewardsCfg:
    """空奖励配置 (Demo用途)"""
    pass

@configclass
class TerminationsCfg:
    """空终止配置 (Demo用途: 不自动重置)"""
    pass


# =============================================================================
# 7. 环境总配置 (Environment Configuration)
# =============================================================================
@configclass
class PipeRobotEnvCfg(ManagerBasedRLEnvCfg):
    # 场景
    scene: PipeRobotSceneCfg = PipeRobotSceneCfg(num_envs=1, env_spacing=2.0)
    # 观测
    observations: ObservationsCfg = ObservationsCfg()
    # 动作
    actions: ActionsCfg = ActionsCfg()
    # 事件
    events: EventCfg = EventCfg()
    # 奖励 (RL必须)
    rewards: RewardsCfg = RewardsCfg()
    # 终止条件 (RL必须)
    terminations: TerminationsCfg = TerminationsCfg()
    # 回合长度 (RL必须)
    episode_length_s: float = 1000.0
    
    # 仿真参数
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.01, # 物理步长 10ms
        render_interval=1, # 每步都渲染，保证流畅交互
    )
    decimation = 1 # 决策频率 = 物理频率


# =============================================================================
# 8. 主程序与键盘交互 (Main & Demo Class)
# =============================================================================
class PiprRobotDemo:
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        # 获取输入接口
        self.input = carb.input.acquire_input_interface()
        # 修改：通过 omni.appwindow 获取键盘
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        # 订阅键盘事件
        self.sub_keyboard = self.input.subscribe_to_keyboard_events(
            self.keyboard, self._on_keyboard_event
        )
        
        # 控制命令状态
        self.wheel_vel_cmd = 0.0  # 轮子目标速度 (rad/s)
        self.arm_pos_cmd = 0.0    # 臂位置偏移 (rad)
        
        # Action Manager 的 active_terms 是一个包含名称的列表 (List of strings)
        self.term_names = self.env.action_manager.active_terms
        print(f"[INFO] Action Terms: {self.term_names}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        """处理键盘回调"""
        # 注意: 这里的 print 可能在终端被刷屏掩盖
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # W: 前进
            if event.input == carb.input.KeyboardInput.W:
                self.wheel_vel_cmd = 5.0 
                print("[CMD] Move Forward")
            # S: 后退
            elif event.input == carb.input.KeyboardInput.S:
                self.wheel_vel_cmd = -5.0
                print("[CMD] Move Backward")
            # U: 大臂下压 (-5度) - 假设负号是下压
            elif event.input == carb.input.KeyboardInput.U:
                self.arm_pos_cmd -= math.radians(5.0) 
                print(f"[CMD] Arm Down: {math.degrees(self.arm_pos_cmd):.1f} deg")
            # I: 大臂抬起 (+5度)
            elif event.input == carb.input.KeyboardInput.I:
                self.arm_pos_cmd += math.radians(5.0) 
                print(f"[CMD] Arm Up: {math.degrees(self.arm_pos_cmd):.1f} deg")
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # 松开 W/S: 停止轮子
            if event.input == carb.input.KeyboardInput.W or event.input == carb.input.KeyboardInput.S:
                self.wheel_vel_cmd = 0.0

        return True

    def run(self):
        """运行主循环"""
        obs, _ = self.env.reset()
        
        # 获取 Action Manager 以便了解动作顺序
        action_mgr = self.env.action_manager
        # 注意：这里的顺序必须与 action tensor 的拼接顺序一致
        # active_terms 是个字典，在 Python 3.7+ 中字典保持插入顺序
        # ActionsCfg 中的定义顺序决定了 tensor 的切片顺序
        
        print("\n" + "="*40)
        print("Pipe Robot Interactive Demo")
        print("Controls:")
        print("  [W] Drive Forward")
        print("  [S] Drive Backward")
        print("  [U] Arm Down (-5 deg)")
        print("  [I] Arm Up   (+5 deg)")
        print("="*40 + "\n")
        
        while simulation_app.is_running():
            # 1. 动态构建 Action Tensor
            # 我们需要把 wheel cmd 和 arm cmd 填入正确的位置
            
            actions_list = []
            
            for term_name in self.term_names:
                # 使用 get_term 方法获取 term 对象
                term = action_mgr.get_term(term_name)
                dim = term.action_dim 
                
                # 创建空白动作
                term_action = torch.zeros((self.env.num_envs, dim), device=self.env.device)
                
                if term_name == "wheels":
                    term_action[:] = self.wheel_vel_cmd
                elif term_name == "arms":
                    # 所有臂关节统一应用偏移
                    term_action[:] = self.arm_pos_cmd
                elif term_name == "steer":
                    # 保持 0
                    pass
                
                actions_list.append(term_action)
            
            # 2. 拼接完整动作
            full_action = torch.cat(actions_list, dim=1)
            
            # 3. 步进环境
            obs, rew, terminated, truncated, info = self.env.step(full_action)
            
            # 4. 自动复位 (虽然 Demo 一般不复位，但这防备跑丢)
            if terminated.any() or truncated.any():
                obs, _ = self.env.reset()


def main():
    # 1. 实例化配置
    env_cfg = PipeRobotEnvCfg()
    # 2. 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 3. 启动交互 Demo
    demo = PiprRobotDemo(env)
    demo.run()
    
    # 4. 退出
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()


