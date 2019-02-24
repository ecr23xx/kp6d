# Object Detector

## Training

Here I use [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) to train a YOLO v3 object detector. Correspondence code in this project located in [darknet](darknet)

### Preparation

First, compile on your system. Currenly this repository only support [Compile on Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux). Darknet also supports compile on Windows. Please refer to [Compile on Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows) if you want to test on Windows. Here I provide one [Makefile](darknet/Makefile), which could compile correctly on my device (Ubuntu 16.04, 4 NVIDIA GTX 1080, CUDA=9.0).

After compilation, run following script to prepare training data for darknet.

```
$ python darknet/scripts/prepare_linemod.py --seq SEQUENCE NUMBER
```

For training details, please read [How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).

### Training

After preparation, run following script to train darknet on LINEMOD.

```
$ sh darknet/scripts/train.sh
```

During training, loss curve will be saved [darknet/chart.jpg](darknet/chart.jpg). You can stop training when seeing loss curve no longer declines. After that, you can run following script to calculate mAP, and choose the weights file with highest mAP.

```
$ sh darknet/scripts/eval.sh
```

## Evaluation

Here I modify [ECer23/yolov3.pytorch](https://github.com/ECer23/yolov3.pytorch) (wrote by myself) for evaluation part.

### Demo

Because object detection part will be integerated into the whole pose estimation pipeline finally, we only provide a demo to test this evaluation part. Run following script, and detection results (images with bounding boxes) will be saved to [eval/results](eval/results)

```
$ python eval/src/demo.py --bs=BATCH SIZE \
                          --reso=RESOLUTION \
                          --gpu=GPU ID \
                          --name=DATASET NAME \
                          --seq=SEQUENCE NUMBER \
                          --ckpt=CHECKPOINTS
```

## Detection result

| Sequence | mAP@0.5 | Recall | Precision | Average IoU |
| -------- | ------- | ------ | --------- | ----------- |
| 01 ape   | 99.9%   | 1.00   | 1.00      | 84.53%      |

