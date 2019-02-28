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

| Sequence | Mean ADD | Mean IoU | Mean 2d Projection Acc |
| -------- | -------- | -------- | ---------------------- |
| 01 Ape   | 0.670    | 1.000    | 0.983                  |
|          |          |          |                        |
|          |          |          |                        |



## Todos

- [x] ~~Keypoint designation~~
- [ ] Data synthesis
- [x] ~~Labels generation~~
- [x] ~~Object detection~~
- [x] ~~Keypoint localization~~
- [x] ~~Pose estimation~~