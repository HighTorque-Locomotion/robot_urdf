import torch
from pyinstrument import Profiler

import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped

import matplotlib.pyplot as plt
import numpy as np


# default_device = "cpu"
default_device = "cuda:0"


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


def plot_3D_xz_with_phase(ax, phase, x_toe, z_toe, _label):
    ax.plot(phase.numpy(), x_toe.numpy(), z_toe.numpy(), label=_label)
    ax.set_title("3D Toe Trajectory")
    ax.set_xlabel("Phase")
    ax.set_ylabel("X Position")
    ax.set_zlabel("Z Position")


class Pai_Cl_cycle:
    # default_device = "cpu"
    default_device = "cuda:0"

    def __init__(self, phase, side="l") -> None:
        self.phase: torch.Tensor = phase
        # print("2",self.phase[0])
        self.side = side
        self.T = torch.tensor(0.3, device=self.default_device)  # 步态周期
        self.dt = 0.01
        self.step = torch.round((self.T / self.dt)).int()
        # print(self.step)
        self.beta = 0.5  # 站姿相位的比例因子
        self.T_P_beta = self.T * self.beta
        self.omega = 0.0  # 角速度
        self.lx = 1.0  # 腿长
        self.k_i = 1.0  # 常数
        self.h = 0.03  # 抬高高度
        self.h_f = self.h/10  # 抬高高度
        self.x_bias = 0.04
        self.ref_dof_pos = torch.zeros(
            (self.phase.size(dim=0), 6), device=self.default_device
        )
        self.foot_end_local = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.default_device
        )
        self.phyi = torch.zeros_like(self.phase, device=self.default_device)
        self.pi = torch.zeros_like(self.phase, device=self.default_device)
        self.vx = torch.ones_like(self.phase, device=self.default_device)  # x 方向速度
        self.vy = torch.zeros_like(self.phase, device=self.default_device)  # y 方向速度
        self.first_step = torch.zeros_like(self.phase, device=self.default_device)
        if self.side == "l":
            self.l0 = [-5.1979e-05, 0.0233, -0.033]
            self.l1 = [0, 0.0568, 0]
            self.l2 = [0, 0, -0.06925]
            self.l3 = [0, 0, -0.07025]
            self.l4 = [0, 0, -0.14]
            self.l5 = [0.07525, 0, 0]
            self.start_phy = (
                0.0 * torch.ones_like(self.phase, device=self.default_device)
            )
        elif self.side == "r":
            self.l0 = [-5.1979e-05, -0.0233, -0.033]
            self.l1 = [0.00025, -0.0568, 0]
            self.l2 = [-0.00025, 0, -0.06925]
            self.l3 = [0, -0.0027, -0.07025]
            self.l4 = [0, 0, -0.14]
            self.l5 = [0.07525, 0.0027, 0]
            self.start_phy = (
                (0.0 - self.T/2)* torch.ones_like(self.phase, device=self.default_device)
            )
        self._l1 = self.l2[2] + self.l3[2]
        self._l1_p = self._l1**2
        self._l2 = self.l4[2]
        self._l2_p = self._l2**2
        self._l1_p_a_2_p = self._l1_p + self._l2_p
        self._l1_2_2 = 2 * self._l1 * self._l2

        self.compute_dict()
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.tfs = TransformStamped()

    def compute(self):
        self.init_pos()
        self.fresh_joint_angle_with_ref()

        self.compute_by_dict(self.start_phy)

        # self.compute_phy(self.start_phy)
        # self.compute_ploy()
        # self.compute_r_i()

        self.fresh_ref_joint_angle_with_cycle()
        # self.fresh_joint_angle_with_ref()

    def init_pos(self):
        self.ref_dof_pos[:, 0] = -0.25
        self.ref_dof_pos[:, 3] = 0.65
        self.ref_dof_pos[:, 4] = -0.4
        self.j0 = self.ref_dof_pos[:, 0]
        self.j3 = self.ref_dof_pos[:, 3]
        self.y = torch.full_like(self.j0, 0.0801)  # y 分量是常数

    def fresh_joint_angle_with_ref(self):
        # 计算 x, y, z, w 分量
        self.x = (
            0.1395 * torch.sin(self.j0)
            + 0.14 * torch.sin(self.j0 + self.j3)
            - 5.1979e-5
        )
        self.z = -0.1395 * torch.cos(self.j0) - 0.14 * torch.cos(self.j0 + self.j3)

    def compute_p_i(self):
        phi_i = (self.phase % self.T) / self.T * 2 * torch.pi
        return phi_i

    def compute_phy(self, start_phase):
        self.phyi = ((self.phase + start_phase) % self.T) / self.T * 2 * torch.pi
        self.phyi[self.phyi > torch.pi] -= 2 * torch.pi
        self.pi = torch.abs(self.phyi) / torch.pi

    def compute_ploy(self):
        pi_cubed = self.pi**3
        pi_fourth = pi_cubed * self.pi
        pi_fifth = pi_fourth * self.pi
        pi_sixth = pi_fifth * self.pi
        self.poly_x_y = 6 * pi_fifth - 15 * pi_fourth + 10 * pi_cubed - 0.5
        self.poly_z = -64 * pi_sixth + 192 * pi_fifth - 192 * pi_fourth + 64 * pi_cubed

    def compute_by_dict(self, start_phase):

        self.indices = torch.round(
            ((self.phase + start_phase) % self.T) / self.dt
        ).int()
        self.indices[self.indices == self.step] = 0
        # print(self.indices)
        self.poly_x_y = self.dict_poly_x_y[self.indices]
        self.poly_z = self.dict_poly_z[self.indices]
        self.r_i_x = self.x + self.vx * self.derta_x[self.indices] - self.x_bias
        self.r_i_z = self.z + self.derta_z[self.indices]
        # self.r_i_z[self.phyi < 0] = self.z[self.phyi < 0]
        # plot_2D(self.phase.clone().cpu(), self.r_i_z.clone().cpu())

    def compute_dict(self):
        phase = torch.arange(
            0, self.T, step=dt, device=self.default_device, dtype=torch.float
        )
        phyi = (phase % self.T) / self.T * 2 * torch.pi
        phyi[phyi > torch.pi] -= 2 * torch.pi
        pi = torch.abs(phyi) / torch.pi

        pi_cubed = pi**3
        pi_fourth = pi_cubed * pi
        pi_fifth = pi_fourth * pi
        pi_sixth = pi_fifth * pi
        self.dict_poly_x_y = 6 * pi_fifth - 15 * pi_fourth + 10 * pi_cubed - 0.5
        self.dict_poly_z = (
            -64 * pi_sixth + 192 * pi_fifth - 192 * pi_fourth + 64 * pi_cubed
        )
        self.derta_x = self.T_P_beta * self.dict_poly_x_y
        self.derta_z = self.h * self.dict_poly_z
        self.derta_z[phyi < 0] = self.h_f * self.dict_poly_z[phyi < 0]
        # plot_2D(phyi.cpu(), self.dict_poly_x_y.cpu())

    def compute_r_i(self):
        self.r_i_x = self.x + self.vx * self.T_P_beta * self.poly_x_y - self.x_bias
        self.r_i_z = self.z + self.h * self.poly_z
        self.r_i_z[self.phyi < 0] = self.z[self.phyi < 0]
        # plot_2D(self.phase.clone().cpu(), self.r_i_z.clone().cpu())

    def fresh_ref_joint_angle_with_cycle(self):
        d = (self.r_i_x**2 + self.r_i_z**2 - self._l1_p_a_2_p) / self._l1_2_2

        sita2 = torch.atan2(torch.sqrt(1 - d**2), d)
        sita2[sita2 < -torch.pi] += 2 * torch.pi
        sita1 = torch.atan2(
            self._l2 * torch.sin(sita2), self._l1 + self._l2 * torch.cos(sita2)
        ) - torch.atan2(self.r_i_x, self.r_i_z)
        sita1[sita1 < -torch.pi] += 2 * torch.pi
        self.ref_dof_pos[:, 0] = -sita1  # + 0.25
        self.ref_dof_pos[:, 3] = sita2  # - 0.65
        self.ref_dof_pos[:, 4] = -sita2 + sita1  # + 0.4

    def pub_tf(self, num):
        self.tfs.header.stamp = rospy.Time.now()
        # print(
        #     self.ref_dof_pos[num, 0], self.ref_dof_pos[num, 3], self.ref_dof_pos[num, 4]
        # )
        self.doPose(
            self.l0,
            self.ref_dof_pos[num, 0],
            "base_link",
            self.side + "_hip_pitch_link",
        )
        self.doPose(
            self.l1,
            self.ref_dof_pos[num, 1],
            self.side + "_hip_pitch_link",
            self.side + "_hip_roll_link",
        )
        self.doPose(
            self.l2,
            self.ref_dof_pos[num, 2],
            self.side + "_hip_roll_link",
            self.side + "_thigh_link",
        )
        self.doPose(
            self.l3,
            self.ref_dof_pos[num, 3],
            self.side + "_thigh_link",
            self.side + "_calf_link",
        )
        self.doPose(
            self.l4,
            self.ref_dof_pos[num, 4],
            self.side + "_calf_link",
            self.side + "_ankle_pitch_link",
        )
        self.doPose(
            self.l5,
            self.ref_dof_pos[num, 5],
            self.side + "_ankle_pitch_link",
            self.side + "_ankle_roll_link",
        )

    def doPose(self, pose, theta, parent, child):
        self.tfs.header.frame_id = parent

        self.tfs.child_frame_id = child
        self.tfs.transform.translation.x = pose[0]
        self.tfs.transform.translation.y = pose[1]
        self.tfs.transform.translation.z = pose[2]
        qtn = tf.transformations.quaternion_from_euler(0, theta.clone().cpu(), 0)
        self.tfs.transform.rotation.x = qtn[0]
        self.tfs.transform.rotation.y = qtn[1]
        self.tfs.transform.rotation.z = qtn[2]
        self.tfs.transform.rotation.w = qtn[3]
        #         4-3.广播器发布数据
        self.broadcaster.sendTransform(self.tfs)


if __name__ == "__main__":
    rospy.init_node("joint_state_publisher")
    nn = 1000
    # 创建一个 Profiler 对象
    profiler = Profiler()
    # 开始性能分析
    profiler.start()
    num_envs = 100
    episode_length_s = 24
    dt = 0.01
    max_episode_length = np.ceil(episode_length_s / dt)
    print("max_episode_length: ", max_episode_length)
    # episode_length_buf = torch.zeros(num_envs, device=default_device, dtype=torch.long)
    episode_length_buf = torch.arange(
        0, num_envs, step=1, device=default_device, dtype=torch.long
    )
    phase = episode_length_buf * 0.01
    l_cyc = Pai_Cl_cycle(phase, "l")
    l_cyc.vx *= 0.4
    r_cyc = Pai_Cl_cycle(phase, "r")
    r_cyc.vx *= 0.4
    ref_dof_pos = torch.zeros(num_envs, 12, device=default_device)
    rate = rospy.Rate(33)

    # l_cyc.compute()
    # plot_2D(l_cyc.phase.cpu(), l_cyc.indices.cpu())
    # print(l_cyc.indices)
    # plot_2D(l_cyc.phase.cpu(), l_cyc.float_indices.cpu())
    # print(l_cyc.phase)
    # t_len = torch.arange(0, l_cyc.T, step=dt, device=default_device, dtype=torch.float)
    # print(t_len)
    # plot_2D(l_cyc.phase.cpu(),l_cyc.poly_z.cpu())
    # plot_2D(l_cyc.phase.cpu(),l_cyc.pi.cpu())
    # plot_2D(l_cyc.phase.cpu(),l_cyc.phyi.cpu())
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # plot_3D_xz_with_phase(
    #     ax, l_cyc.phase.cpu(), l_cyc.pi.cpu(), l_cyc.phyi.cpu(), "left"
    # )
    # r_cyc.compute()
    # plot_2D(r_cyc.phyi.cpu(),r_cyc.pi.cpu())
    # l_cyc.pub_tf(0)
    # r_cyc.pub_tf(0)
    # a = l_cyc.ref_dof_pos
    # print("l_cyc.ref_dof_pos: ",
    #     a[0, 0].cpu(),
    #     a[0, 3].cpu(),
    #     a[0, 4].cpu(),
    # )
    # plot_2D(
    #     l_cyc.phase.cpu(),
    #     l_cyc.ref_dof_pos[:, 0].cpu(),
    #     ylabel="hip_pitch",
    #     _label="hip_pitch",
    # )
    # plot_2D(
    #     l_cyc.phase.cpu(),
    #     l_cyc.ref_dof_pos[:, 3].cpu(),
    #     ylabel="calf_pitch",
    #     _label="calf_pitch",
    # )
    # plot_2D(
    #     l_cyc.phase.cpu(),
    #     l_cyc.ref_dof_pos[:, 4].cpu(),
    #     ylabel="ankle_pitch",
    #     _label="ankle_pitch",
    # )

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # plot_3D_xz_with_phase(
    #     ax, l_cyc.phase.cpu(), l_cyc.r_i_x.cpu(), l_cyc.r_i_z.cpu(), "left"
    # )

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # plot_3D_xz_with_phase(
    #     ax, r_cyc.phase.cpu(), r_cyc.r_i_x.cpu(), r_cyc.r_i_z.cpu(), "right"
    # )

    plt.show()
    while not rospy.is_shutdown():
        episode_length_buf += 1
        phase = episode_length_buf * 0.01
        # print(phase[0], (phase[0] % l_cyc.T) / l_cyc.T * 2 * torch.pi)
        l_cyc.phase = phase.clone()
        r_cyc.phase = phase.clone()
        l_cyc.compute()
        l_cyc.pub_tf(0)

        r_cyc.compute()
        r_cyc.pub_tf(0)

        ref_dof_pos[:, 0] = l_cyc.ref_dof_pos[:, 0]
        ref_dof_pos[:, 3] = l_cyc.ref_dof_pos[:, 3]
        ref_dof_pos[:, 4] = l_cyc.ref_dof_pos[:, 4]

        ref_dof_pos[:, 6] = r_cyc.ref_dof_pos[:, 0]
        ref_dof_pos[:, 9] = r_cyc.ref_dof_pos[:, 3]
        ref_dof_pos[:, 10] = r_cyc.ref_dof_pos[:, 4]
        nn -= 1
        if not nn:
            break
        rate.sleep()
    # 停止性能分析
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

    # 生成 HTML 报告
    profiler.open_in_browser()
