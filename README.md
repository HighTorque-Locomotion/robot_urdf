如果你要自主添加realsense相机，请在catkin_ws/src 下放 realsense2_description 包。

仓库位于`git clone https://github.com/IntelRealSense/realsense-ros.git -b ros1-legacy`

使用 `roslaunch pai_xxxxxx display_with_camera.launch` 即可。

需要调整相机位置，请在 `src/xxxx/urdf/pai_with_camera.urdf.xacro` 中修改`origin`标签的数值
```
<xacro:sensor_d435i parent="base_link" use_nominal_extrinsics="$(arg use_nominal_extrinsics)">
    <origin xyz="0.12 0 0.11" rpy="0 0 0"/>
</xacro:sensor_d435i>
```



