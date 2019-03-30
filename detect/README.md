# Object Detector

## Training

Training code is based on [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), which located in [darknet/](darknet)

### Installation

First, compile on your system. Currenly this repository only support [Compile on Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux). Darknet also supports compile on Windows. Please refer to [Compile on Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows) if you want to test on Windows. Here I provide one [Makefile](darknet/Makefile), which could compile correctly on my device (Ubuntu 16.04, 4 NVIDIA GTX 1080, CUDA=9.0).

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

## Evaluation Demo

Here I modify [ECer23/yolov3.pytorch](https://github.com/ECer23/yolov3.pytorch) (wrote by myself) for evaluation part.

Because object detection part will be integerated into the whole pose estimation pipeline finally, we only provide a [demo.py](/demo.py) to test this evaluation part. Run following script, and detection results (images with bounding boxes) will be saved to [eval/results/](eval/results)

```
$ python detect.py --bs=BATCH SIZE \
                   --reso=RESOLUTION \
                   --gpu=GPU ID \
                   --name=DATASET NAME \
                   --seq=SEQUENCE NUMBER \
                   --ckpt=CHECKPOINTS
```

## Detection result

| Sequence       | AP@0.5 | Recall | Precision | F1-score | Average IoU | Step |
| -------------- | ------ | ------ | --------- | -------- | ----------- | ---- |
| 01 ape         | 100.0% | 1.00   | 1.00      | 1.00     | 92.02%      | 4000 |
| 02 benchvise   | 100.0% | 1.00   | 1.00      | 1.00     | 92.26%      | 4500 |
| 04 camera      | 100.0% | 1.00   | 1.00      | 1.00     | 91.95%      | 3700 |
| 05 can         | 100.0% | 1.00   | 1.00      | 1.00     | 90.88%      | 2500 |
| 06 cat         | 100.0% | 1.00   | 1.00      | 1.00     | 89.31%      | 2700 |
| 08 driller     | 100.0% | 1.00   | 1.00      | 1.00     | 90.62%      | 3200 |
| 09 duck        | 100.0% | 1.00   | 1.00      | 1.00     | 92.07%      | 3800 |
| 12 holepuncher | 100.0% | 1.00   | 1.00      | 1.00     | 90.12%      | 2500 |



