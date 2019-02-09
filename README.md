# kp6d

Gao PENG's undergraduate final year project.

## Table of Contents

- [Todos](#todo)
- [Pipeline](#pipeline)
    - [Keypoints Designation](#keypoints-designation)

## Todos

- [x] ~~Keypoint designation~~
- [ ] Data synthesis
- [ ] Labels generation
- [ ] Object detection
    - [x] ~~Training~~
    - [ ] Evaluation
- [ ] Keypoint localization
- [ ] Pose estimation
- [ ] Combine together

## Pipeline

### Keypoints Designation

Go to folder [gendata](./gendata) and follow its instructions. After this step, you'll get models and keypoints raw data (in .ply format).

### Object Detector Training

Go to folder [yolo](./yolo) and follow its instructions. After this step, you'll get pre-trained weights for LINEMOD.

