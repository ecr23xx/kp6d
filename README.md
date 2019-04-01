# kp6d

My undergraduate final year project.

## Table of Contents

- [Pipeline](#pipeline)
    - [Prepare Data](#prepare-data)
    - [Keypoints designation](#keypoints-designation)
    - [Object detector](#object-detector)
    - [Keypoint Localization](#keypoint-localization)
    - [Pose Estimation](#pose-estimation)
    - [Evaluation](#evaluation)
- [Results](#results)
- [Todos](#todos)


## Pipeline

### Prepare Data

Go to folder [data/](./data) and follow its instructions. After this step, data will be prepared for training.

### Keypoints Designation

Go to folder [gendata/](./gendata) and follow its instructions. After this step, you'll get models and keypoints raw data (in .ply format).

### Object Detector

Go to folder [detect/](./detect) and follow its instructions. After this step, you'll get YOLOv3 pre-trained weights for LINEMOD.

### Keypoint Localization

Go to folder [keypoint/](./keypoint) and follow its instructions. After this step, you'll get SPPE pre-trained weights for LINEMOD.

### Pose Estimation

Run [scripts/pose.sh](./scripts/pose.sh) to estimate the pose. Results will be saved to [results/](results/).

### Evaluation

Run [scripts/eval.sh](scripts/eval.sh) to estimate the pose and evaluate the result

## Results

### 17 SIFT keypoints

| Sequence       | Mean ADD Acc  | Mean 2d Projection Acc | Backbone   |
| -------------- | ------------- | ---------------------- | ---------- |
| 01 Ape         | 0.456         | 0.990                  | ResNet-50  |
| 02 Benchvise   | 0.971 (99.9)  | 0.984                  | ResNet-50  |
| 04 Camera      | 0.775 (86.86) | 0.986                  | ResNet-101 |
| 05 Can         | 0.863 (95.47) | 0.981                  |            |
| 06 Cat         | 0.780 (79.34) | 0.990                  | ResNet-50  |
| 08 Driller     | 0.867 (96.43) | 0.952                  | ResNet-101 |
| 09 Duck        | 0.532         | 0.988                  | ResNet-50  |
| 10 Eggbox      | 0.907 (99.15) | 0.985                  | ResNet-50  |
| 11 Glue        | 0.851 (95.66) | 0.985                  |            |
| 12 Holepuncher | 0.654 (81.92) | 0.992                  |            |
| 13 Iron        | 0.955 (98.88) | 0.983                  |            |
| 14 Lamp        | 0.949 (99.33) | 0.973                  |            |
| 15 Phone       | 0.738 (92.41) | 0.981                  | ResNet-50  |
| Average        | (0.89)        |                        |            |

### 17 Cluster keypoints

| Sequence       | Mean ADD Acc | Mean 2d Projection Acc | Backbone  |
| -------------- | ------------ | ---------------------- | --------- |
| 01 Ape         | 0.649        | 0.991                  | ResNet-50 |
| 02 Benchvise   | 0.985 (99.9) | 0.997                  | ResNet-50 |
| 04 Camera      | 0.886        | 0.986                  | ResNet-50 |
| 05 Can         | 0.922 (95.5) | 0.981                  | ResNet-50 |
| 06 Cat         | 0.866        | 0.990                  | ResNet-50 |
| 08 Driller     | 0.963 (96.4) | 0.979                  | ResNet-50 |
| 09 Duck        | 0.588        | 0.988                  | ResNet-50 |
| 10 Eggbox      | (99.2)       | 0.994                  |           |
| 11 Glue        | (95.7)       | 0.985                  |           |
| 12 Holepuncher | 0.763 (81.9) | 0.996                  | ResNet-50 |
| 13 Iron        | 0.967 (98.9) | 0.994                  | ResNet-50 |
| 14 Lamp        | 0.981 (99.3) | 0.985                  | ResNet-50 |
| 15 Phone       | 0.865 (92.4) | 0.990                  | ResNet-50 |
| Average        | 0.726 (0.89) |                        |           |


## Todos

- [x] ~~Keypoint designation~~
- [ ] Data synthesis
- [x] ~~Labels generation~~
- [x] ~~Object detection~~
- [x] ~~Keypoint localization~~
- [x] ~~Pose estimation~~

