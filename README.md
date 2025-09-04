## 说明

作为hightorque model最新修订版本集合，每次要修改urdf或者xml，大家进行讨论，然后把最新修改的同步到这里，这样大家主要用这个就好了；

## 修改日志

**2025.09.04：**新增小hi 25dof xml文件。

![image-20250904151249019](./README.assets/image-20250904151249019.png)

**2025.08.28：**将小pi初始位置归为竖直。

训练时，default pos可以参考：

```python
    # 设置机器人默认关节角度
    default_joint_angles = [
        -0.25,  # r_hip_pitch_joint
        0.0,    # r_hip_roll_joint
        0.0,    # r_thigh_joint
        0.65,   # r_calf_joint
        -0.4,   # r_ankle_pitch_joint
        0.0,    # r_ankle_roll_joint
        -0.25,  # l_hip_pitch_joint
        -0.0,   # l_hip_roll_joint
        0.0,    # l_thigh_joint
        0.65,   # l_calf_joint
        -0.4,   # l_ankle_pitch_joint
        0.0     # l_ankle_roll_joint
    ]
```



**2025.09.03： **将小hi 25自由度urdf上传 ，无xml；pi plus 24dof urdf上传，其中xml是20dof的。



