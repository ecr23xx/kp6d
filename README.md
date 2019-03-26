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

| Sequence       | Mean ADD Acc | Mean 2d Projection Acc |
| -------------- | ------------ | ---------------------- |
| 01 Ape         |              |                        |
| 02 Benchvise   | 0.965        | 0.984                  |
| 04 Camera      | 0.791        | 0.986                  |
| 05 Can         | 0.766        | 0.971                  |r
| 06 Cat         | 0.660        | 0.986                  |
| 08 Driller     | 0.857        | 0.955                  |
| 09 Duck        | 0.471        | 0.987                  |
| 10 Eggbox      | 0.671        | 0.994                  |
| 11 Glue        | 0.598        | 0.985                  |
| 12 Holepuncher | 0.637        | 0.993                  |
| 13 Iron        | 0.955        | 0.983                  |
| 14 Lamp        | 0.949        | 0.973                  |
| Average        |              |                        |



## Todos

- [x] ~~Keypoint designation~~
- [ ] Data synthesis
- [x] ~~Labels generation~~
- [x] ~~Object detection~~
- [x] ~~Keypoint localization~~
- [x] ~~Pose estimation~~