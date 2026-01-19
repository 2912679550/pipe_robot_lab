import torch
import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp

# =============================================================================
# 自定义动作项 (Custom Actions)
# =============================================================================

# 定义一些常量
MAIN_WHEEL_R    = 0.05 # 主轮半径 (米), 直径100mm
ASSIST_WHEEL_R  = 0.03 # 辅助轮半径 (米)，直径60mm

@configclass
class LinkedArmActionCfg(mdp.JointPositionActionCfg):
    """Configuration for LinkedArmAction."""
    class_type = None # Set below
    # 基础配置
    asset_name: str = "robot"
    # 控制mid_arm的关节 (同时也是 joint_names)
    joint_names: list[str] = [".*mid_arm_.*"] 
    # 跟随联动的tail_arm关节（会自动计算）
    tail_joint_names: list[str] = [".*tail_arm_.*"]
    def __post_init__(self):
        # 显式设置 implementation class
        self.class_type = LinkedArmAction
        super().__post_init__()
class LinkedArmAction(mdp.JointPositionAction):
    """自定义联动动作：控制mid_arm，自动计算tail_arm"""
    cfg: LinkedArmActionCfg
    def __init__(self, cfg: LinkedArmActionCfg, env: ManagerBasedRLEnv):
        # 初始化基类，只注册 mid_arm 关节 (由 cfg.joint_names 指定)
        super().__init__(cfg, env)
        # 获取tail_arm关节的索引
        asset = env.scene[cfg.asset_name]
        tail_joint_names = []
        for pattern in cfg.tail_joint_names:
            # find_joints 返回 (indices, names)
            ids, _ = asset.find_joints(pattern)
            tail_joint_names.extend(ids)
        # 存储tail_arm关节的索引
        self.tail_joint_idxs = torch.tensor(tail_joint_names, device=env.device, dtype=torch.int32)
        # mid_arm索引存储在 self._joint_ids (基类处理)
        print(f"[INFO] LinkedArmAction: {len(self._joint_ids)} mid_arms -> {len(self.tail_joint_idxs)} tail_arms")

        # --- Debug: 打印关节名称配对以进行验证 ---
        # 获取实际的关节名称
        # 注意: 如果 _joint_ids 是 tensor，需先转为 list
        mid_ids_list = self._joint_ids.tolist() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        tail_ids_list = self.tail_joint_idxs.tolist() if isinstance(self.tail_joint_idxs, torch.Tensor) else self.tail_joint_idxs
        
        mid_names = [asset.joint_names[i] for i in mid_ids_list]
        tail_names = [asset.joint_names[i] for i in tail_ids_list]
        
        print(f"[INFO] LinkedArmAction Pairing Check ({len(mid_names)} pairs):")
        # 打印配对情况，便于用户检查是否错位（如 mid_arm_01 对应到了 tail_arm_02）
        for i, (m, t) in enumerate(zip(mid_names, tail_names)):
            print(f"  Pair {i:02d}: {m:<30} --> {t}")
        
        if len(mid_names) != len(tail_names):
             print(f"[WARNING] LinkedArmAction: Mismatch in number of joints! Mid: {len(mid_names)}, Tail: {len(tail_names)}")
    def apply_actions(self):
        # 1. 应用 mid_arm (基类逻辑)
        # process_actions 已经在 step 前被调用，self.processed_actions 包含 mid_arm 的目标位置
        super().apply_actions()
        mid_targets = self.processed_actions
        # 计算并配置 tail targets
        tail_targets = self._compute_tail_position(mid_targets)
        self._asset.set_joint_position_target(tail_targets, joint_ids=self.tail_joint_idxs)
    def _compute_tail_position(self, mid_positions: torch.Tensor) -> torch.Tensor:
        """根据mid_arm位置计算tail_arm位置的自定义函数"""
        AB = 26.88
        BC = 151.5
        CD = 37
        DA = 162
        # 四连杆机构，mid对应角ABC， tail对应(pi - angleBCD)
        # 角ABC的实际值等于 mid_arm + 57.63°， tail为0时， (pi - angleBCD) = 47.87°
        # 计算 mid_arm 对应的 angleABC
        angle_ABC = mid_positions + math.radians(57.63)
        # 使用余弦定理计算 对角线 AC 的长度
        AC = torch.sqrt(AB**2 + BC**2 - 2*AB*BC*torch.cos(angle_ABC))
        # 进一步使用余弦定理分别计算 angleBCA和angele ACD
        angle_BCA = torch.acos((BC**2 + AC**2 - AB**2) / (2 * BC * AC))
        angle_ACD = torch.acos((AC**2 + CD**2 - DA**2) / (2 * AC * CD))
        # 计算 angleBCD
        angle_BCD = angle_BCA + angle_ACD
        # 计算 tail_arm 位置 (tail_positions)
        tail_positions = math.pi - angle_BCD
        # 使用余弦定理计算 angleBCD
        # 转换为相对于默认位置的偏移量
        return tail_positions - math.radians(47.87)

@configclass
class SteerWheelActionCfg(mdp.JointVelocityActionCfg):
    # 配置一组舵轮联合控制
    asset_name: str = "robot"
    # 轮电机
    joint_names: list[str] = [".*wheel_.*"]
    # 与轮电机配套对应的舵电机
    steer_joint_names: list[str] = [".*steer_.*"]
    scale: float = 1.0
    def __post_init__(self):
        self.class_type = SteerWheelAction
        super().__post_init__()
    
class SteerWheelAction(mdp.JointVelocityAction):
    cfg: SteerWheelActionCfg
    def __init__(self , cfg: SteerWheelActionCfg, env: ManagerBasedRLEnv):
        # 1. 初始化基类
        super().__init__(cfg, env)
        # 2. 获取配套的舵关节索引
        asset = env.scene[cfg.asset_name]
        steer_joint_names = []
        for pattern in cfg.steer_joint_names:
            ids, _ = asset.find_joints(pattern)
            steer_joint_names.extend(ids)

        # 3. 存储索引
        # self._joint_ids 存储的是 Wheel 的索引 (基类管理)
        # self.steer_joint_idxs 存储的是 Steer 的索引
        self.steer_joint_idxs = torch.tensor(steer_joint_names, device=env.device, dtype=torch.int32)
        
        # --- Debug Info ---
        wheel_ids = self._joint_ids.tolist() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        steer_ids = self.steer_joint_idxs.tolist() if isinstance(self.steer_joint_idxs, torch.Tensor) else self.steer_joint_idxs
        
        print(f"[DEBUG] SteerWheelAction Init: WheelPattern={cfg.joint_names}, SteerPattern={cfg.steer_joint_names}")
        print(f"        -> Found Wheels: {wheel_ids}, Steers: {steer_ids}")
        
        if len(wheel_ids) == 0:
            print(f"[ERROR] No WHEEL joints found matching {cfg.joint_names}")
        if len(steer_ids) == 0:
            print(f"[ERROR] No STEER joints found matching {cfg.steer_joint_names}")

        # 确保轮子数量和舵数量一致
        if len(self._joint_ids) != len(self.steer_joint_idxs) and len(wheel_ids) > 0:
            raise ValueError(f"Number of wheels ({len(self._joint_ids)}) and steers ({len(self.steer_joint_idxs)}) must match!")
        # 4. 打印配对信息
        # 修正：当只有一个索引且为 list 类型时 (基类 behavior 可能不同)，不要使用 .tolist()
        # 基类 mdp.JointVelocityAction 的 _joint_ids 可能是 torch.Tensor 也可能是 list
        # 安全地将其转换为 list
        wheel_ids = self._joint_ids.tolist() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        steer_ids = self.steer_joint_idxs.tolist() if isinstance(self.steer_joint_idxs, torch.Tensor) else self.steer_joint_idxs
        
        wheel_names = [asset.joint_names[i] for i in wheel_ids]
        steer_names = [asset.joint_names[i] for i in steer_ids]
        
        if len(wheel_names) > 0 and len(steer_names) > 0:
             print(f"[INFO] SteerWheelAction Unit: {wheel_names[0]} <--> {steer_names[0]}")

    @property
    def action_dim(self):
        # 覆盖动作维度：虽然控制 N 个轮子，但输入只有 2 维 (Vx, Vy)
        # 这个动作将广播给通过 regex 匹配到的所有轮子
        return 2
    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # process_actions 通常用于处理 clip 或 scale
        # 这里的 actions 输入就是 (Env_Num, 2) 的 [Vx, Vy]
        # 我们暂时只做简单的缩放，实际解算在 apply_actions 中进行
        return actions * self.cfg.scale
    def apply_actions(self):
        # Input: [Vx, Vy] -> (Num_Envs, 2)
        cmds = self.processed_actions
        vx = cmds[:, 0]
        vy = cmds[:, 1]
        
        # 1. 运动解算 (Swerve Kinematics)
        # 轮速 V = sqrt(vx^2 + vy^2)
        target_speed = torch.hypot(vx, vy).unsqueeze(1) # (Num_Envs, 1)
        
        # 2. 舵角计算
        # 只有当速度大于阈值时，才更新目标角度
        # (Isaac Sim中保持上次Command即保持位置)
        # 但如果是 position control，我们需要明确的值。
        # 简单方案：总是计算，0速度时归零。复杂方案：保留上一帧。
        # 这里为了防止“松手归零”，我们如果速度极小，就不改变Steer的目标
        # 但是 Action 必须是一个 Tensor。
        
        # 当前策略：正常计算，不特殊处理（先排除故障，再谈优化）
        target_angle = torch.atan2(vy, vx).unsqueeze(1) # (Num_Envs, 1)
        
        # 3. 下发指令 (注意：只下发给单一关节)
        # 保护：防止空索引导致错误 (虽然上面init检查了，但为了健壮性)
        if len(self._joint_ids) > 0:
            self._asset.set_joint_velocity_target(target_speed, joint_ids=self._joint_ids)
        if len(self.steer_joint_idxs) > 0:
            # TODO: 如果需要保持角度，可在此处加入逻辑
            self._asset.set_joint_position_target(target_angle, joint_ids=self.steer_joint_idxs)


# =============================================================================
# 3. 动作配置 (Actions Configuration)
# =============================================================================
@configclass
class ActionsCfg:
    # -------------------------------------------------------------------------
    # 舵轮阵列 (Swerve Drive Modules)
    # 请根据实际关节名称，复制 6 份并修改正则
    # -------------------------------------------------------------------------
    # 使用 SteerWheelActionCfg，每个实例仅控制一对 [Wheel, Steer]
    steer_wheel_01 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_01"],        # 左后辅助轮
        steer_joint_names=[".*assist_steer_01"],  # 左后辅助轮舵向
        # 通过scale将线速度转换为角速度： wheel_ang_vel = line_vel / wheel_radius
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_02 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_02"],        # 右后辅助轮
        steer_joint_names=[".*assist_steer_02"],  # 右后辅助轮
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_03 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*main_wheel_01"],          # 中后主动轮
        steer_joint_names=[".*main_steer_01"],    # 中后主动轮舵向
        scale=1.0 / MAIN_WHEEL_R,
    )
    steer_wheel_04 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_03"],          # 左前辅助轮
        steer_joint_names=[".*assist_steer_03"],    # 左前辅助轮
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_05 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_04"],          # 右前辅助轮
        steer_joint_names=[".*assist_steer_04"],    # 右前辅助轮
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_06 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*main_wheel_02"],          # 中前主动轮
        steer_joint_names=[".*main_steer_02"],    # 中前主动轮
        scale=1.0 / MAIN_WHEEL_R,
    )
    # -------------------------------------------------------------------------
    # ! 机械臂与弯折 (Manipulators)
    # -------------------------------------------------------------------------
    # * 联动臂 实际控制关节数量: 2 * 4 = 8
    # 后侧两组联动臂
    pipe_dia_01 = LinkedArmActionCfg(
        asset_name="robot",
        joint_names=["mid_arm_01", "mid_arm_02"],
        tail_joint_names=["tail_arm_01", "tail_arm_02"],
        scale=1.0,
        use_default_offset=True,
    )
    # 前侧两组联动臂
    pipe_dia_02 = LinkedArmActionCfg(
        asset_name="robot",
        joint_names=["mid_arm_03", "mid_arm_04"],
        tail_joint_names=["tail_arm_03", "tail_arm_04"],
        scale=1.0,
        use_default_offset=True,
    )
    # * 两组大臂控制 实际控制关节数量: 2 * 2 = 4
    up_arms_01 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*up_arm_01", ".*up_arm_02"],
        scale=1.0,
        use_default_offset=True, 
    )
    up_arms_02 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*up_arm_03", ".*up_arm_04"],
        scale=1.0,
        use_default_offset=True, 
    )
    # * 机身弯折控制
    bend_01 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["bend_01"],
        scale=1.0,
        use_default_offset=True,
    )
    bend_02 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["bend_02"],
        scale=1.0,
        use_default_offset=True,
    )
    # =============================================================================

