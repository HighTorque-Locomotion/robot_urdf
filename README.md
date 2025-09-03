## 作为hightorque model最新修订版本集合，每次要修改urdf或者xml，大家进行讨论，然后把最新修改的同步到这里，这样大家主要用这个就好了；

## 修改了pi、hi的xml文件，增加了joint参数：damping\stifness\armature,主要是armature这个参数，目前设置的是0.01888,不一定准确，星动纪元和宇树科技都是0.02；

## 修改了pi的xml脚部碰撞模型，和urdf一致；



## 2025.8.28将小pi初始位置归为竖直状态

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



## 2025.9.3 将小hi 25自由度urdf上传 ，无xml，pi plus 24dof urdf上传，xml是20dof的

