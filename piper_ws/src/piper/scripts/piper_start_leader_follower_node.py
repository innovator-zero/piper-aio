#!/usr/bin/env python3
# Leader-follower Piper node.
# mode=0 publishes both leader and follower arm state to ROS.
# mode=1 controls the follower arm from leader topics.
# DAgger support: /teach/* topics allow runtime switching between teach mode
# (human drags leader, follower mirrors) and follower-control mode (policy controls).
import math
import threading
import time

import rosnode
import rospy
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PiperStatusMsg, PosCmd
from piper_sdk import *
from piper_sdk import C_PiperInterface
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger, TriggerResponse
from tf.transformations import quaternion_from_euler


def check_roscore():
    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("roscore is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("roscore is not running.")
        raise RuntimeError("roscore is not running.")


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


class C_PiperRosNode:
    """Piper ROS node."""

    def __init__(self) -> None:
        check_roscore()
        rospy.init_node("piper_start_all_node", anonymous=True)

        self.can_port = "can0"
        if rospy.has_param("~can_port"):
            self.can_port = rospy.get_param("~can_port")
            rospy.loginfo("%s is %s", rospy.resolve_name("~can_port"), self.can_port)
        else:
            rospy.loginfo("can_port parameter not found, please use _can_port:=can0.")
            exit(0)
        self.mode = 0
        if rospy.has_param("~mode"):
            self.mode = rospy.get_param("~mode")
            rospy.loginfo("%s is %s", rospy.resolve_name("~mode"), self.mode)
        else:
            rospy.loginfo("mode parameter not found, please use _mode:=0.")
            exit(0)

        self.dagger_leader = False
        if rospy.has_param("~dagger_leader"):
            self.dagger_leader = _to_bool(rospy.get_param("~dagger_leader")) and self.mode == 0
        rospy.loginfo("%s is %s", rospy.resolve_name("~dagger_leader"), self.dagger_leader)

        self.auto_enable = False
        if rospy.has_param("~auto_enable"):
            if _to_bool(rospy.get_param("~auto_enable")) and self.mode == 1:
                self.auto_enable = True
        rospy.loginfo("%s is %s", rospy.resolve_name("~auto_enable"), self.auto_enable)
        self.gripper_exist = True

        # DAgger teach-mode state. This is active only for dedicated 4CAN
        # leader nodes launched with ~dagger_leader:=true.
        self.is_enabled = False
        self.current_linkage_config = 0xFC
        self.in_teach_mode = False
        # publish
        self.joint_std_pub_follower = rospy.Publisher(
            "/follower/joint_states", JointState, queue_size=1, tcp_nodelay=True
        )
        if self.mode == 0:
            self.joint_std_pub_leader = rospy.Publisher(
                "/leader/joint_states", JointState, queue_size=1, tcp_nodelay=True
            )
        self.arm_status_pub = rospy.Publisher("/follower/arm_status", PiperStatusMsg, queue_size=1, tcp_nodelay=True)
        self.end_pose_pub = rospy.Publisher("/follower/end_pose", PoseStamped, queue_size=1, tcp_nodelay=True)
        self.end_pose_euler_pub = rospy.Publisher("/follower/end_pose_euler", PosCmd, queue_size=1, tcp_nodelay=True)
        self.__enable_flag = False
        self.joint_state_follower = JointState()
        self.joint_state_follower.name = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.joint_state_follower.position = [0.0] * 7
        self.joint_state_follower.velocity = [0.0] * 7
        self.joint_state_follower.effort = [0.0] * 7
        self.joint_state_leader = JointState()
        self.joint_state_leader.name = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.joint_state_leader.position = [0.0] * 7
        self.joint_state_leader.velocity = [0.0] * 7
        self.joint_state_leader.effort = [0.0] * 7

        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()

        if self.dagger_leader:
            self._configure_dagger_linkage(self.current_linkage_config)

        # service
        str_can_port = str(self.can_port)
        # Leader arm home service.
        self.leader_go_zero_service = rospy.Service(
            "/" + str_can_port + "/go_zero_leader",
            Trigger,
            self.handle_leader_go_zero_service,
        )
        # Leader and follower arm home service.
        self.leader_follower_go_zero_service = rospy.Service(
            "/" + str_can_port + "/go_zero_leader_follower",
            Trigger,
            self.handle_leader_follower_go_zero_service,
        )
        # Restore leader-follower mode service.
        self.restore_leader_follower_mode_service = rospy.Service(
            "/" + str_can_port + "/restore_leader_follower_mode",
            Trigger,
            self.handle_restore_leader_follower_mode_service,
        )
        if self.mode == 1:
            sub_pos_th = threading.Thread(target=self.SubPosThread)
            sub_joint_th = threading.Thread(target=self.SubJointThread)
            sub_enable_th = threading.Thread(target=self.SubEnableThread)

            sub_pos_th.daemon = True
            sub_joint_th.daemon = True
            sub_enable_th.daemon = True

            sub_pos_th.start()
            sub_joint_th.start()
            sub_enable_th.start()

        if self.dagger_leader:
            teach_ctrl_th = threading.Thread(target=self.SubTeachControlThread)
            teach_ctrl_th.daemon = True
            teach_ctrl_th.start()

    def GetEnableFlag(self):
        return self.__enable_flag

    def Pubilsh(self):
        """Publish arm messages."""
        rate = rospy.Rate(200)  # 200 Hz
        enable_flag = False
        timeout = 5
        start_time = time.time()
        elapsed_time_flag = False
        while not rospy.is_shutdown():
            if self.auto_enable and self.mode == 1:
                while not (enable_flag):
                    elapsed_time = time.time() - start_time
                    print("--------------------")
                    enable_flag = (
                        self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                        and self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                        and self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                        and self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                        and self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                        and self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
                    )
                    print("Enable Status:", enable_flag)
                    if enable_flag:
                        self.__enable_flag = True
                        self.is_enabled = True
                    self.piper.EnableArm(7)
                    self.piper.GripperCtrl(0, 1000, 0x02, 0)
                    self.piper.GripperCtrl(0, 1000, 0x01, 0)
                    print("--------------------")
                    if elapsed_time > timeout:
                        print("timeout....")
                        elapsed_time_flag = True
                        enable_flag = True
                        break
                    time.sleep(1)
                    pass
            if elapsed_time_flag:
                print("auto enable timeout, exiting program")
                exit(0)
            self.PublishFollowerArmJointAndGripper()
            self.PublishFollowerArmState()
            self.PublishFollowerArmEndPose()
            if self.mode == 0 and (not self.dagger_leader or self.in_teach_mode):
                self.PublishLeaderArmJointAndGripper()

            rate.sleep()

    def PublishFollowerArmState(self):
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        )
        arm_status.communication_status_joint_2 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        )
        arm_status.communication_status_joint_3 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        )
        arm_status.communication_status_joint_4 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        )
        arm_status.communication_status_joint_5 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        )
        arm_status.communication_status_joint_6 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        )
        self.arm_status_pub.publish(arm_status)

    def PublishFollowerArmEndPose(self):
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        endpos.header.stamp = rospy.Time.now()
        self.end_pose_pub.publish(endpos)

        end_pose_euler = PosCmd()
        end_pose_euler.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        end_pose_euler.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        end_pose_euler.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        end_pose_euler.roll = roll
        end_pose_euler.pitch = pitch
        end_pose_euler.yaw = yaw
        end_pose_euler.gripper = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        end_pose_euler.mode1 = 0
        end_pose_euler.mode2 = 0
        self.end_pose_euler_pub.publish(end_pose_euler)

    def PublishFollowerArmJointAndGripper(self):
        self.joint_state_follower.header.stamp = rospy.Time.now()
        joint_0: float = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        joint_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        vel_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
        self.joint_state_follower.position = [
            joint_0,
            joint_1,
            joint_2,
            joint_3,
            joint_4,
            joint_5,
            joint_6,
        ]  # Example values
        self.joint_state_follower.velocity = [
            vel_0,
            vel_1,
            vel_2,
            vel_3,
            vel_4,
            vel_5,
            0.0,
        ]  # Example values
        self.joint_state_follower.effort[6] = effort_6
        self.joint_std_pub_follower.publish(self.joint_state_follower)

    def PublishLeaderArmJointAndGripper(self):
        self.joint_state_leader.header.stamp = rospy.Time.now()
        joint_0: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_6 / 1000) * 0.017444
        joint_6: float = self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle / 1000000
        # Fallback: in teach mode GetArmJointCtrl() may return zeros if the CAN ctrl
        # channel is inactive. Use feedback positions instead (matches dagger reference).
        if self.in_teach_mode and abs(joint_0) < 0.001 and abs(joint_1) < 0.001 and abs(joint_2) < 0.001:
            rospy.logwarn_throttle(1, "GetArmJointCtrl returned zeros in teach mode, using feedback positions")
            joint_0 = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
            joint_1 = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
            joint_2 = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
            joint_3 = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
            joint_4 = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
            joint_5 = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
            joint_6 = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        self.joint_state_leader.position = [
            joint_0,
            joint_1,
            joint_2,
            joint_3,
            joint_4,
            joint_5,
            joint_6,
        ]  # Example values
        self.joint_std_pub_leader.publish(self.joint_state_leader)

    def SubTeachControlThread(self):
        """Subscribe to 4CAN DAgger leader runtime control topics."""
        rospy.Subscriber("/teach/enable", Bool, self.teach_enable_callback, queue_size=1)
        rospy.Subscriber("/teach/linkage_config", String, self.teach_linkage_config_callback, queue_size=1)
        rospy.Subscriber("/teach/teach_mode", Int32, self.teach_mode_callback, queue_size=1)
        rospy.Subscriber(
            "/leader_cmd/joint_states", JointState,
            self.controlled_joint_callback, queue_size=1, tcp_nodelay=True,
        )
        rospy.spin()

    def _configure_dagger_linkage(self, linkage_config):
        """Configure 4CAN leader linkage without changing enable state."""
        try:
            self.piper.MasterSlaveConfig(
                linkage_config=linkage_config,
                feedback_offset=0x00,
                ctrl_offset=0x00,
                linkage_offset=0x00,
            )
            self.current_linkage_config = linkage_config
            rospy.loginfo("DAgger leader: linkage configured to 0x%02X", linkage_config)
            time.sleep(2)
        except Exception as e:
            rospy.logerr("DAgger leader linkage init failed: %s", e)

    def teach_enable_callback(self, msg):
        """Enable or disable the arm (used when entering/exiting DAgger mode)."""
        if not self.dagger_leader:
            return
        try:
            if msg.data and not self.is_enabled:
                self.piper.EnableArm(7)
                self.is_enabled = True
                self.__enable_flag = True
                rospy.loginfo("DAgger: arm enabled")
                time.sleep(2)
                if self.current_linkage_config == 0xFA:
                    self._dagger_init_teach_mode()
                else:
                    self._dagger_init_follower_control_mode()
            elif not msg.data and self.is_enabled:
                self.piper.DisableArm(7)
                self.is_enabled = False
                self.__enable_flag = False
                rospy.loginfo("DAgger: arm disabled")
        except Exception as e:
            rospy.logerr("DAgger enable callback failed: %s", e)

    def teach_linkage_config_callback(self, msg):
        """Switch arm between teach mode (0xFA) and follower-control/CAN-ctrl mode (0xFC)."""
        if not self.dagger_leader:
            return
        config_map = {"leader": 0xFA, "follower": 0xFC, "0xfa": 0xFA, "0xfc": 0xFC, "fa": 0xFA, "fc": 0xFC}
        config_str = msg.data.lower().strip()
        new_config = config_map.get(config_str)
        if new_config is None:
            rospy.logwarn("DAgger linkage_config: unknown value '%s'", msg.data)
            return
        if new_config == self.current_linkage_config:
            return
        try:
            rospy.loginfo("DAgger: switching linkage config 0x%02X -> 0x%02X", self.current_linkage_config, new_config)
            self.piper.MasterSlaveConfig(
                linkage_config=new_config, feedback_offset=0x00, ctrl_offset=0x00, linkage_offset=0x00,
            )
            time.sleep(2)
            if self.is_enabled:
                self.piper.EnableArm(7)
                time.sleep(2)
            self.current_linkage_config = new_config
            if new_config == 0xFA:
                self._dagger_init_teach_mode()
            else:
                self._dagger_init_follower_control_mode()
            rospy.loginfo("DAgger: linkage config switched to 0x%02X", new_config)
        except Exception as e:
            rospy.logerr("DAgger linkage_config callback failed: %s", e)

    def teach_mode_callback(self, msg):
        """Enter (1) or exit (0) drag-teach mode."""
        if not self.dagger_leader:
            return
        try:
            if msg.data == 1:
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)
                self.in_teach_mode = True
                rospy.loginfo("DAgger: entered drag-teach mode")
            else:
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)
                self.in_teach_mode = False
                time.sleep(0.5)
                self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30)
                rospy.loginfo("DAgger: exited drag-teach mode, CAN control active")
        except Exception as e:
            rospy.logerr("DAgger teach_mode callback failed: %s", e)

    def controlled_joint_callback(self, joint_data):
        """Execute joint commands sent while arm is in follower-control (CAN-ctrl) mode.

        Only applies to 4CAN leader nodes while they are in 0xFC CAN-control
        mode. Follower nodes are commanded exclusively by joint_callback.
        """
        if not self.dagger_leader:
            return
        if self.in_teach_mode:
            return
        if len(joint_data.position) < 7:
            rospy.logwarn("DAgger leader command ignored short JointState: %d positions", len(joint_data.position))
            return
        factor = 57324.840764  # 1000 * 180 / pi
        joint_0 = round(joint_data.position[0] * factor)
        joint_1 = round(joint_data.position[1] * factor)
        joint_2 = round(joint_data.position[2] * factor)
        joint_3 = round(joint_data.position[3] * factor)
        joint_4 = round(joint_data.position[4] * factor)
        joint_5 = round(joint_data.position[5] * factor)
        joint_6 = round(joint_data.position[6] * 1_000_000)
        joint_6 = max(0, min(80000, joint_6))
        if self.GetEnableFlag():
            self.piper.MotionCtrl_2(0x01, 0x01, 30)
            self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            if self.gripper_exist:
                self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x01, 30)

    def _dagger_init_teach_mode(self):
        """Put arm into drag-teach mode (called after switching to 0xFA)."""
        if self.is_enabled:
            self.piper.EnableArm(7)
            time.sleep(2)
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)
            self.in_teach_mode = True
            rospy.loginfo("DAgger: arm in drag-teach mode, human can drag now")

    def _dagger_init_follower_control_mode(self):
        """Put arm back into CAN-control mode (called after switching to 0xFC)."""
        if self.is_enabled:
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)
            self.in_teach_mode = False
            time.sleep(1)
            self.piper.EnableArm(7)
            time.sleep(2)
            self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30)
            rospy.loginfo("DAgger: arm in CAN-control mode")

    def SubPosThread(self):
        """Subscribe to end-effector pose commands."""
        rospy.Subscriber("/pos_cmd", PosCmd, self.pos_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def SubJointThread(self):
        """Subscribe to joint commands."""
        rospy.Subscriber(
            "/leader/joint_states",
            JointState,
            self.joint_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.spin()

    def SubEnableThread(self):
        """Subscribe to enable commands."""
        rospy.Subscriber("/enable_flag", Bool, self.enable_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def pos_callback(self, pos_data):
        """Handle end-effector pose commands."""
        factor = 180 / 3.1415926
        x = round(pos_data.x * 1000) * 1000
        y = round(pos_data.y * 1000) * 1000
        z = round(pos_data.z * 1000) * 1000
        rx = round(pos_data.roll * 1000 * factor)
        ry = round(pos_data.pitch * 1000 * factor)
        rz = round(pos_data.yaw * 1000 * factor)
        rospy.loginfo("Received PosCmd:")
        rospy.loginfo("x: %f", x)
        rospy.loginfo("y: %f", y)
        rospy.loginfo("z: %f", z)
        rospy.loginfo("roll: %f", rx)
        rospy.loginfo("pitch: %f", ry)
        rospy.loginfo("yaw: %f", rz)
        rospy.loginfo("gripper: %f", pos_data.gripper)
        rospy.loginfo("mode1: %d", pos_data.mode1)
        rospy.loginfo("mode2: %d", pos_data.mode2)
        if self.GetEnableFlag():
            self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)
            self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)
            gripper = round(pos_data.gripper * 1000 * 1000)
            if pos_data.gripper > 80000:
                gripper = 80000
            if pos_data.gripper < 0:
                gripper = 0
            if self.gripper_exist:
                self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)

    def joint_callback(self, joint_data):
        """Handle joint commands."""
        factor = 57324.840764  # 1000*180/3.14
        rospy.loginfo("Received Joint States:")
        rospy.loginfo("joint_0: %f", joint_data.position[0] * 1)
        rospy.loginfo("joint_1: %f", joint_data.position[1] * 1)
        rospy.loginfo("joint_2: %f", joint_data.position[2] * 1)
        rospy.loginfo("joint_3: %f", joint_data.position[3] * 1)
        rospy.loginfo("joint_4: %f", joint_data.position[4] * 1)
        rospy.loginfo("joint_5: %f", joint_data.position[5] * 1)
        rospy.loginfo("joint_6: %f", joint_data.position[6] * 1)
        joint_0 = round(joint_data.position[0] * factor)
        joint_1 = round(joint_data.position[1] * factor)
        joint_2 = round(joint_data.position[2] * factor)
        joint_3 = round(joint_data.position[3] * factor)
        joint_4 = round(joint_data.position[4] * factor)
        joint_5 = round(joint_data.position[5] * factor)
        joint_6 = round(joint_data.position[6] * 1000 * 1000)
        if joint_6 > 80000:
            joint_6 = 80000
        if joint_6 < 0:
            joint_6 = 0
        if self.GetEnableFlag():
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            pass

    def enable_callback(self, enable_flag: Bool):
        """Handle enable commands."""
        rospy.loginfo("Received enable flag:")
        rospy.loginfo("enable_flag: %s", enable_flag.data)
        if enable_flag.data:
            self.__enable_flag = True
            self.is_enabled = True
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x02, 0)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.__enable_flag = False
            self.is_enabled = False
            self.piper.DisableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x00, 0)

    def handle_leader_go_zero_service(self, req):
        response = TriggerResponse()
        rospy.loginfo(f"-----------------------RESET---------------------------")
        rospy.loginfo(f"{self.can_port} send piper leader go zero service")
        rospy.loginfo(f"-----------------------RESET---------------------------")
        self.piper.ReqMasterArmMoveToHome(1)
        response.success = True
        response.message = str({self.can_port}) + "send piper leader go zero service success"
        rospy.loginfo(f"Returning resetResponse: {response.success}, {response.message}")
        return response

    def handle_leader_follower_go_zero_service(self, req):
        response = TriggerResponse()
        rospy.loginfo(f"-----------------------RESET---------------------------")
        rospy.loginfo(f"{self.can_port} send piper leader follower go zero service")
        rospy.loginfo(f"-----------------------RESET---------------------------")
        self.piper.ReqMasterArmMoveToHome(2)
        response.success = True
        response.message = str({self.can_port}) + "send piper leader follower go zero service success"
        rospy.loginfo(f"Returning resetResponse: {response.success}, {response.message}")
        return response

    def handle_restore_leader_follower_mode_service(self, req):
        response = TriggerResponse()
        rospy.loginfo(f"-----------------------RESET---------------------------")
        rospy.loginfo(f"{self.can_port} send piper restore leader follower mode service")
        rospy.loginfo(f"-----------------------RESET---------------------------")
        self.piper.ReqMasterArmMoveToHome(0)
        response.success = True
        response.message = str({self.can_port}) + "send piper restore leader follower mode service success"
        rospy.loginfo(f"Returning resetResponse: {response.success}, {response.message}")
        return response


if __name__ == "__main__":
    try:
        piper_leader_follower = C_PiperRosNode()
        piper_leader_follower.Pubilsh()
    except rospy.ROSInterruptException:
        pass
