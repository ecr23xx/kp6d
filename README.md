# kp6d

My undergraduate final year project.

## Table of Contents

- [Todos](#todos)
- [Pipeline](#pipeline)
    - [Prepare Data](#prepare-data)
    - [Keypoints designation](#keypoints-designation)
    - [Object detector](#object-detector)
    - [Keypoint Localization](#keypoint-localization)

## Todos

- [x] ~~Keypoint designation~~
- [ ] Data synthesis
- [x] ~~Labels generation~~
- [x] ~~Object detection~~
- [x] ~~Keypoint localization~~
- [ ] Pose estimation
- [ ] Combine together

## Pipeline

### Prepare Data

Go to folder [data/](./data) and follow its instructions. After this step, data will be prepared for training.

### Keypoints Designation

Go to folder [gendata/](./gendata) and follow its instructions. After this step, you'll get models and keypoints raw data (in .ply format).

### Object Detector

Go to folder [detect/](./detect) and follow its instructions. After this step, you'll get YOLOv3 pre-trained weights for LINEMOD.

### Keypoint Localization

Go to folder [keypoint/](./keypoint) and follow its instructions. After this step, you'll get SPPE pre-trained weights for LINEMOD.

