# RayVO

## Framework of RayVO

<!-- ![Alt text](visualization/framework_rayvo.png)
<img src="https://raw.githubusercontent.com/fanqixucs/RayVO/main/visualization/framework_rayvo.png" width="500">  -->

<div align="center">
  <img src="https://raw.githubusercontent.com/fanqixucs/RayVO/main/visualization/framework_rayvo.png" width="600">
</div>

This picture shows the framework of Ray Visual Odometry: (1) Training process: For each image in the pair of images, we denoise the noisy GT rays, either in inverse or forward pose, conditioned on their features from DINOv2 to predict the rays for forward pose and inverse pose. (2) Inference process: During the inference time, given two ray bundles, inverse and forward, we first calculate the corresponding Lie algebra (6-DoF) for them. Then, the final predicted 6-DoF VO is calculated as the average of these two poses.


## Video of RayVO
<div align="center">
  <img src="https://raw.githubusercontent.com/fanqixucs/RayVO/main/visualization/RayVO_video_trajectory_visualization.gif" width="600" />
</div>

<!--
[![Watch the video](https://raw.githubusercontent.com/fanqixucs/RayVO/main/visualization/framework_rayvo.png)](visualization/video_trajectory_ray_representation.mp4)
-->
