import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
import torch
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped


def rt_mat(theta, d):
    """
    生成一个二维平面上的旋转和平移矩阵
    theta: 旋转角度（弧度）
    d: 平移向量 (dx, dy)
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    dx, dy, dz = d
    return torch.tensor(
        [
            [cos_theta, 0, -sin_theta, dx],
            [0, 1, 0, dy],
            [sin_theta, 0, cos_theta, dz],
            [0, 0, 0, 1],
        ]
    )


def plot_3D_xz_with_phase(ax, phase, x_toe, z_toe, _label):
    ax.plot(phase.numpy(), x_toe.numpy(), z_toe.numpy(), label=_label)
    ax.set_title("3D Toe Trajectory")
    ax.set_xlabel("Phase")
    ax.set_ylabel("X Position")
    ax.set_zlabel("Z Position")


def plot_2D(
    x,
    y,
    _label="Toe Trajectory",
    title="Toe Trajectory",
    xlable="Phase",
    ylabel="X Position",
):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=_label)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    plt.grid(True)
    # plt.show()


def leg(phase, sin_pos, cos_pos, side=1):
    scale_1 = 0.3  # 假设的 scale_1 值
    scale_2 = 2 * scale_1
    l0 = (0, side * 0.0233, -0.033)
    l1 = (0, 0, -0.1395)
    l2 = (0, 0, -0.14)
    l3 = (0.05, 0, 0)
    ref_dof_pos = torch.zeros((1000, 12))
    ref_dof_pos[:, 0] = sin_pos * scale_1
    ref_dof_pos[:, 3] = -sin_pos * scale_2
    ref_dof_pos[:, 4] = sin_pos * (scale_2 - scale_1)
    # ref_dof_pos[torch.abs(sin_pos) < 0.3] = 0
    ref_dof_pos[:, 0] = ref_dof_pos[:, 0] + 0.25
    ref_dof_pos[:, 3] = ref_dof_pos[:, 3] - 0.65
    ref_dof_pos[:, 4] = ref_dof_pos[:, 4] + 0.4

    rt_mat_hip_pitch = torch.stack([rt_mat(a, l0) for a in ref_dof_pos[:, 0]])
    rt_mat_calf = torch.stack([rt_mat(a, l1) for a in ref_dof_pos[:, 3]])
    rt_mat_ankle_pitch = torch.stack([rt_mat(a, l2) for a in ref_dof_pos[:, 4]])
    rt_mat_ankle_roll = torch.stack([rt_mat(a, l3) for a in ref_dof_pos[:, 5]])
    total_rt_mat = (
        rt_mat_hip_pitch @ rt_mat_calf @ rt_mat_ankle_pitch @ rt_mat_ankle_roll
    )
    # print(total_rt_mat.size())
    foot_end_local = torch.tensor([0.0, 0.0, 0.0, 1.0])

    hip_pitch_base = (rt_mat_hip_pitch @ foot_end_local)[:, :3]
    knee_pitch_base = (rt_mat_hip_pitch @ rt_mat_calf @ foot_end_local)[:, :3]
    ankle_pitch_base = (
        rt_mat_hip_pitch @ rt_mat_calf @ rt_mat_ankle_pitch @ foot_end_local
    )[:, :3]
    foot_end_base = (
        rt_mat_hip_pitch
        @ rt_mat_calf
        @ rt_mat_ankle_pitch
        @ rt_mat_ankle_roll
        @ foot_end_local
    )[:, :3]

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_ylim(-0.3, 0.3)
    # ax.set_zlim(-0.3, 0.3)
    # plot_3D_xz_with_phase(ax,phase,hip_pitch_base[:,0],hip_pitch_base[:,2],"hip_pitch")
    # plot_3D_xz_with_phase(ax,phase,knee_pitch_base[:,0],knee_pitch_base[:,2],"knee_pitch")
    # plot_3D_xz_with_phase(ax,phase,ankle_pitch_base[:,0],ankle_pitch_base[:,2],"ankle_pitch")
    # plot_3D_xz_with_phase(ax,phase,foot_end_base[:,0],foot_end_base[:,2],"foot_end")
    # ax.legend()
    # plt.figure(figsize=(10, 6))
    dist = torch.norm(knee_pitch_base - hip_pitch_base, dim=1).numpy()
    # print(dist)
    # plt.plot(phase.numpy(), dist, label="Toe Trajectory")
    # plt.title("Toe Trajectory")
    # plt.xlabel("Phase")
    # plt.ylabel("X Position")
    # plt.grid(True)
    # plt.show()
    return (hip_pitch_base, knee_pitch_base, ankle_pitch_base, foot_end_base)

num_envs = 4096

class Leg:
    hip_pitch_base: torch.Tensor
    hip_roll_base: torch.Tensor
    thigh_base: torch.Tensor
    calf_base: torch.Tensor
    ankle_pitch_base: torch.Tensor
    ankle_roll_base: torch.Tensor

    def __init__(
        self,
        phase: torch.Tensor,
        ax,
        side="l",
        scale=0.3,
    ) -> None:
        self.phase: torch.Tensor = phase
        self.ax = ax
        # 定义参数
        self.T = 0.5  # 步态周期
        self.beta = 0.5  # 站姿相位的比例因子
        self.vy = 0.0  # y 方向速度
        self.omega = 0.0  # 角速度
        self.lx = 1.0  # 腿长
        self.k_i = 1.0  # 常数
        self.h = 0.03
        self.ref_dof_pos = torch.zeros((self.phase.size(dim=0), 6))
        self.vx = 0.3*torch.ones_like(self.phase)  # x 方向速度
        self.scale_1 = scale  # 假设的 scale_1 值
        self.scale_2 = 2 * self.scale_1
        self.init_pos()

        if side == "l":
            self.l0 = [-5.1979e-05, 0.0233, -0.033]
            self.l1 = [0, 0.0568, 0]
            self.l2 = [0, 0, -0.06925]
            self.l3 = [0, 0, -0.07025]
            self.l4 = [0, 0, -0.14]
            self.l5 = [0.07525, 0, 0]
            self.sin_pos = torch.sin(2 * torch.pi * self.phase)
            self.cos_pos = torch.cos(2 * torch.pi * self.phase)
            self.fresh_joint_angle_with_ref()

            # self.fresh_ref_joint_angle_with_sin()
            self.compute_r_i(0*torch.pi)
            self.fresh_ref_joint_angle_with_cycle()
            (self.line,) = ax.plot([], [], [], lw=2, label="left")
        elif side == "r":
            self.l0 = [-5.1979e-05, -0.0233, -0.033]
            self.l1 = [0.00025, -0.0568, 0]
            self.l2 = [-0.00025, 0, -0.06925]
            self.l3 = [0, -0.0027, -0.07025]
            self.l4 = [0, 0, -0.14]
            self.l5 = [0.07525, 0.0027, 0]
            self.sin_pos = torch.sin(2 * torch.pi * self.phase + torch.pi)
            self.cos_pos = torch.cos(2 * torch.pi * self.phase + torch.pi)
            
            self.fresh_joint_angle_with_ref()

            # self.fresh_ref_joint_angle_with_sin()
            self.compute_r_i(torch.pi)
            self.fresh_ref_joint_angle_with_cycle()
            (self.line,) = ax.plot([], [], [], lw=2, label="right")
        print(self.phase.size())

        self.fresh_joint_angle_with_ref()

    def init_pos(self):
        self.ref_dof_pos[:, 0] = -0.25
        self.ref_dof_pos[:, 3] = 0.65
        self.ref_dof_pos[:, 4] = -0.4

    def fresh_joint_angle_with_ref(self):
        rt_mat_hip_pitch = torch.stack(
            [rt_mat(a, self.l0) for a in self.ref_dof_pos[:, 0]]
        )
        rt_mat_hip_roll = torch.stack(
            [rt_mat(a, self.l1) for a in self.ref_dof_pos[:, 1]]
        )
        rt_mat_thigh = torch.stack([rt_mat(a, self.l2) for a in self.ref_dof_pos[:, 2]])
        rt_mat_calf = torch.stack([rt_mat(a, self.l3) for a in self.ref_dof_pos[:, 3]])
        rt_mat_ankle_pitch = torch.stack(
            [rt_mat(a, self.l4) for a in self.ref_dof_pos[:, 4]]
        )
        rt_mat_ankle_roll = torch.stack(
            [rt_mat(a, self.l5) for a in self.ref_dof_pos[:, 5]]
        )
        total_rt_mat = (
            rt_mat_hip_pitch @ rt_mat_calf @ rt_mat_ankle_pitch @ rt_mat_ankle_roll
        )
        # print(total_rt_mat.size())
        foot_end_local = torch.tensor([0.0, 0.0, 0.0, 1.0])

        self.hip_pitch_base = (rt_mat_hip_pitch @ foot_end_local)[:, :3]
        self.hip_roll_base = (rt_mat_hip_pitch @ rt_mat_hip_roll @ foot_end_local)[
            :, :3
        ]
        self.thigh_base = (
            rt_mat_hip_pitch @ rt_mat_hip_roll @ rt_mat_thigh @ foot_end_local
        )[:, :3]
        self.calf_base = (
            rt_mat_hip_pitch
            @ rt_mat_hip_roll
            @ rt_mat_thigh
            @ rt_mat_calf
            @ foot_end_local
        )[:, :3]
        self.ankle_pitch_base = (
            rt_mat_hip_pitch
            @ rt_mat_hip_roll
            @ rt_mat_thigh
            @ rt_mat_calf
            @ rt_mat_ankle_pitch
            @ foot_end_local
        )[:, :3]
        self.ankle_roll_base = (
            rt_mat_hip_pitch
            @ rt_mat_hip_roll
            @ rt_mat_thigh
            @ rt_mat_calf
            @ rt_mat_ankle_pitch
            @ rt_mat_ankle_roll
            @ foot_end_local
        )[:, :3]

    def fresh_ref_joint_angle_with_sin(self):
        self.ref_dof_pos[:, 0] += self.sin_pos * self.scale_1
        self.ref_dof_pos[:, 3] += -self.sin_pos * self.scale_2
        self.ref_dof_pos[:, 4] += self.sin_pos * (self.scale_2 - self.scale_1)

    def compute_p_i(self):
        phi_i = (self.phase % self.T) / self.T * 2 * torch.pi
        return phi_i

    def compute_r_i(self, start_phase=torch.pi):
        phyi = self.compute_p_i() + start_phase
        phyi[phyi > torch.pi] -= 2 * torch.pi
        pi = phyi.clone()
        # pi[phyi < 0] = -phyi[phyi < 0] / torch.pi
        # pi[phyi > 0] = phyi[phyi > 0] / torch.pi
        
        pi = torch.abs(phyi) / torch.pi
        a_i_x = self.vx * self.T * self.beta
        a_i_y = (self.vy + self.k_i * self.omega * self.lx / 2) * self.T * self.beta

        add_x = a_i_x * (6 * pi**5 - 15 * pi**4 + 10 * pi**3 - 0.5)
        add_y = a_i_y * (6 * pi**5 - 15 * pi**4 + 10 * pi**3 - 0.5)
        add_z = self.h * (-64 * pi**6 + 192 * pi**5 - 192 * pi**4 + 64 * pi**3)
        print(add_x.max(),add_x.min())
        self.r_i_x = self.ankle_pitch_base[:, 0] + add_x
        self.r_i_y = self.ankle_pitch_base[:, 1] + add_y
        self.r_i_z = self.ankle_pitch_base[:, 2] - self.l0[2] + add_z
        self.r_i_z[phyi < 0] = self.ankle_pitch_base[phyi < 0, 2] - self.l0[2]
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        plot_3D_xz_with_phase(ax, self.phase, self.r_i_x, self.r_i_z, "r_i_x r_i_z")
        # plot_2D(self.phase, self.r_i_x, ylabel="r_i_x", _label="r_i_x")
        # plot_2D(self.phase, self.r_i_z, ylabel="r_i_z", _label="r_i_z")

    def fresh_ref_joint_angle_with_cycle(self):
        print("fresh_ref_joint_angle_with_cycle")
        print(self.r_i_x.size())
        l1 = self.l2[2] + self.l3[2]
        l2 = self.l4[2]

        d = (self.r_i_x**2 + self.r_i_z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        sita2 = torch.atan2(torch.sqrt(1 - d**2), d)
        sita2[sita2 < -torch.pi] += 2 * torch.pi
        print(sita2[:5])
        # plot_2D(self.phase, sita2, ylabel="sita calf", _label="angle 2")

        sita1 = torch.atan2(
            l2 * torch.sin(sita2), l1 + l2 * torch.cos(sita2)
        ) - torch.atan2(self.r_i_x, self.r_i_z)
        sita1[sita1 < -torch.pi] += 2 * torch.pi

        # plot_2D(self.phase, -sita1, ylabel="sita thigh", _label="angle 1")
        # self.ref_dof_pos[:, 0] = -0.25#-sita1
        self.ref_dof_pos[:, 0] = -sita1
        # self.ref_dof_pos[:, 3] = 0.65#sita2
        self.ref_dof_pos[:, 3] = sita2
        # self.ref_dof_pos[:, 4] = -0.4#-sita2 + sita1
        self.ref_dof_pos[:, 4] = -sita2 + sita1

    def init_line(self):
        self.line.set_data([], [])
        self.line.set_3d_properties([])

    def update_lin(self, num):
        x = [
            self.hip_pitch_base[num, 0],
            self.hip_roll_base[num, 0],
            self.thigh_base[num, 0],
            self.calf_base[num, 0],
            self.ankle_pitch_base[num, 0],
            self.ankle_roll_base[num, 0],
        ]
        z = [
            self.hip_pitch_base[num, 2],
            self.hip_roll_base[num, 2],
            self.thigh_base[num, 2],
            self.calf_base[num, 2],
            self.ankle_pitch_base[num, 2],
            self.ankle_roll_base[num, 2],
        ]
        # print(self.phase[num])
        self.line.set_data(
            [
                self.phase[num],
                self.phase[num],
                self.phase[num],
                self.phase[num],
                self.phase[num],
                self.phase[num],
            ],
            z,
        )
        self.line.set_3d_properties(x)


class plot_two_leg:
    def __init__(self, phase: torch.Tensor) -> None:
        self.phase: torch.Tensor = phase.clone()
        print("1")
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.l_leg = Leg(phase=self.phase, ax=self.ax, side="l", scale=0)
        self.r_leg = Leg(phase=self.phase, ax=self.ax, side="r", scale=0)
        self.pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.tfs = TransformStamped()
        self.init_ros_pub_msg()

        self.set_ax()

    def set_ax(self):
        # 设置轴的范围
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(-0.35, 0.05)
        self.ax.set_zlim(-0.2, 0.2)

        # 设置标签
        self.ax.set_xlabel("Y Label")  # y
        self.ax.set_ylabel("Z Label")
        self.ax.set_zlabel("X Label")  # x

        ani = FuncAnimation(
            self.fig,
            self.update_line,
            frames=len(self.phase),
            init_func=self.init_line,
            blit=False,
            repeat=False,
            interval=50,
        )

        # 显示图形
        plt.show()

    def init_line(self):
        self.l_leg.init_line()
        self.r_leg.init_line()
        return (self.l_leg.line, self.r_leg.line)

    def update_line(self, num):
        self.l_leg.update_lin(num)
        self.r_leg.update_lin(num)
        # self.talker(num)
        self.pub_tf(num)
        return (self.l_leg.line, self.r_leg.line)

    def init_ros_pub_msg(
        self,
    ):
        self.hello_str = JointState()
        self.hello_str.header = Header()
        self.hello_str.header.stamp = rospy.Time.now()
        self.hello_str.name = [
            "r_hip_pitch_joint",
            "r_calf_joint",
            "r_ankle_pitch_joint",
            "l_hip_pitch_joint",
            "l_calf_joint",
            "l_ankle_pitch_joint",
        ]
        self.hello_str.velocity = [0] * (6)
        self.hello_str.effort = [0] * (6)
        self.hello_str.position = [0] * (6)

    def talker(self, num):
        self.hello_str.header.stamp = rospy.Time.now()
        self.hello_str.position[0] = self.r_leg.ref_dof_pos[num, 0]
        self.hello_str.position[1] = self.r_leg.ref_dof_pos[num, 3]
        self.hello_str.position[2] = self.r_leg.ref_dof_pos[num, 4]

        self.hello_str.position[3] = self.l_leg.ref_dof_pos[num, 0]
        self.hello_str.position[4] = self.l_leg.ref_dof_pos[num, 3]
        self.hello_str.position[5] = self.l_leg.ref_dof_pos[num, 4]

        self.pub.publish(self.hello_str)

    def pub_tf(self, num):
        self.tfs.header.stamp = rospy.Time.now()
        self.doPose(
            self.l_leg.l0,
            self.l_leg.ref_dof_pos[num, 0],
            "base_link",
            "l_hip_pitch_link",
        )
        self.doPose(
            self.l_leg.l1,
            self.l_leg.ref_dof_pos[num, 1],
            "l_hip_pitch_link",
            "l_hip_roll_link",
        )
        self.doPose(
            self.l_leg.l2,
            self.l_leg.ref_dof_pos[num, 2],
            "l_hip_roll_link",
            "l_thigh_link",
        )
        self.doPose(
            self.l_leg.l3, self.l_leg.ref_dof_pos[num, 3], "l_thigh_link", "l_calf_link"
        )
        self.doPose(
            self.l_leg.l4,
            self.l_leg.ref_dof_pos[num, 4],
            "l_calf_link",
            "l_ankle_pitch_link",
        )
        self.doPose(
            self.l_leg.l5,
            self.l_leg.ref_dof_pos[num, 5],
            "l_ankle_pitch_link",
            "l_ankle_roll_link",
        )

        self.doPose(
            self.r_leg.l0,
            self.r_leg.ref_dof_pos[num, 0],
            "base_link",
            "r_hip_pitch_link",
        )
        self.doPose(
            self.r_leg.l1,
            self.r_leg.ref_dof_pos[num, 1],
            "r_hip_pitch_link",
            "r_hip_roll_link",
        )
        self.doPose(
            self.r_leg.l2,
            self.r_leg.ref_dof_pos[num, 2],
            "r_hip_roll_link",
            "r_thigh_link",
        )
        self.doPose(
            self.r_leg.l3, self.r_leg.ref_dof_pos[num, 3], "r_thigh_link", "r_calf_link"
        )
        self.doPose(
            self.r_leg.l4,
            self.r_leg.ref_dof_pos[num, 4],
            "r_calf_link",
            "r_ankle_pitch_link",
        )
        self.doPose(
            self.r_leg.l5,
            self.r_leg.ref_dof_pos[num, 5],
            "r_ankle_pitch_link",
            "r_ankle_roll_link",
        )

    def doPose(self, pose, theta, parent, child):
        self.tfs.header.frame_id = parent

        self.tfs.child_frame_id = child
        self.tfs.transform.translation.x = pose[0]
        self.tfs.transform.translation.y = pose[1]
        self.tfs.transform.translation.z = pose[2]
        qtn = tf.transformations.quaternion_from_euler(0, theta, 0)
        self.tfs.transform.rotation.x = qtn[0]
        self.tfs.transform.rotation.y = qtn[1]
        self.tfs.transform.rotation.z = qtn[2]
        self.tfs.transform.rotation.w = qtn[3]
        #         4-3.广播器发布数据
        self.broadcaster.sendTransform(self.tfs)


if __name__ == "__main__":
    rospy.init_node("joint_state_publisher")
    phase = torch.linspace(0, 4, 1000)
    aa = plot_two_leg(phase)
