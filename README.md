
# A Stable Dishware Grasp Synthesis Method based on Contact-GraspNet for Table Bussing Robots


<p align="center">
  <img src="https://github.com/ji-eun-lab/KIST-GraspNet/assets/73579127/3e94ac09-4181-4e79-93e8-03bcea8f8829">
</p>



In this paper, we propose a stable dishware grasp synthesis method for table bussing robots. In order to generate safe grasp candidates that do not collide with other dishware on the table. We first employ Contact-GraspNet using 3D point cloud data, which is one of effective grasp selection methods. The grasp candidates are then filtered and refined to obtain the final grasp information, which helps the robot gripper lift the dishes straight up without spilling any leftover food. In addition, it is implemented and verified in a simulation environment using Isaac Sim, and can be easily considered in the design stage of table bussing robots.


# 연구 방법
1. YOLOv8의 기반의 식기류 instance segmentation
2. Point cloud에 대한 Contact–GraspNet 기반의 파지 후보 집합 생성
3. 파지 후보 집합에 대한 filtering 및 refinement
	- Segmentation mask를 grid map으로 표현하여 내부에 생성된 contact point 제거
	- Baseline 벡터와 contact point 에서 물체 중심으로 향하는 벡터 간 유사도가 높은 contact point 선별
	- Baseline 벡터를 테이블 평면과 평행하게 보정하고, 접근 벡터를 중력방향과 일치시켜 최종 파지 정보 생성

<p align="center">
  <img src="https://github.com/ji-eun-lab/KIST-GraspNet/assets/73579127/14a3d0dc-1b75-4d2e-a242-ae8a9bcb2ea8" >
</p>

<p align="center">
  <img src="https://github.com/ji-eun-lab/KIST-GraspNet/assets/73579127/3369e890-ed9f-49b4-a15b-24fc20f5b79a" width="500" height="300">
</p>


