# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import KINOVA_JACO2_N6S300_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class KinovaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Kinova
        self.scene.robot = KINOVA_JACO2_N6S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # initally do not override events
        # self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # override rewards
        # note "gripper" not working so replacing with "j2n6s300_end_effector"
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["j2n6s300_end_effector"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            "j2n6s300_end_effector"
        ]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["j2n6s300_end_effector"]

        # disable orientation tracking
        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["j2n6s300_joint_[1-6]"], scale=0.5, use_default_offset=True
        )

        # override command generator body
        # end-effector is along ???-direction
        # must be changed based on director of end effector
        self.commands.ee_pose.body_name = "j2n6s300_end_effector"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class KinovaReachEnvCfg_PLAY(KinovaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
