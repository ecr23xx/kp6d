# Keypoint Localizer

Keypoint localizer part is based on [MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).

## Training

### Environment

1. Disable cudnn for `batch_norm`. See [Microsoft/human-pose-estimation.pytorch#installation](https://github.com/Microsoft/human-pose-estimation.pytorch#installation)
    ```bash
    $ export PYTORCH=path/to/pytorch
    $ sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    ```
2. Install [ncullen93/torchsample](https://github.com/ncullen93/torchsample) under [train_sppe/](./train_sppe/)
    ```bash
    $ pip install -e git+https://github.com/ncullen93/torchsample.git#egg=torchsample
    ```

For more details, please refer to [Alphapose Train SPPE Installation Instructions](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch/train_sppe#installation).

### Prepare Data

Go to [prepare_data/](./prepare_data/) folder, run

```bash
$ python prepare_gt.py --seq=SEQ \
                       --kptype=KPTYPE \
                       --kpnum=KPNUM
```

Explanations for arguments:

* `--seq`: Sequence number for LINEMOD.
* `--kptype`: Type of keypoints. Now we only support SIFT keypoints.
* `--kpnum`: Number of keypoints. Now we only support 17 keypoints.

### Train the Model

Train SPPE by running

```bash
python train.py --nClasses=NCLASSES \
                --seq=SEQ \
                --kptype=KPTYPE \
                --optMethod=OPTMETHOD \
                --trainBatch=TRAINBATCH \
                --dataset=DATASET \
                --datatype=DATATYPE \
                --loadModel=LOADMODEL \
                --addDPG \
```

Explanations for arguments:

* `--nClasses`: Number of keypoints.
* `--seq`: Sequence number for LINEMOD.
* `--kptype`: Type of keypoints. Now we only support SIFT keypoints
* `--optmethod`: Optimization methods.
* `--trainBatch`: Training batch
* `--dataset`: Dataset name. Now we only support LINEMOD
* `--datatype`: Ground truth or synthesis data. Now we only support ground truth data.
* `--loadModel`: Load checkpoints from given path, optional.
* `--addDPG`: Add DPG or not, optional.

### Training Details

For every sequence

1. Train without `--addDPG`.
2. Resume training from best model from last step with `--addDPG`.

## Evaluation Demo

### Demo

Because object detection part will be integerated into the whole pose estimation pipeline finally, we only provide a [demo.py](/demo.py) to test this evaluation part. Run following script, and detection results (images with bounding boxes) will be saved to [eval/results/](eval/results)

```
cd /home/penggao/projects/pose/kp6d/
python keypoint.py --bs=BATCH SIZE \
                   --reso=RESOLUTION \
                   --gpu=GPU ID \
                   --name=DATASET NAME \
                   --seq=SEQUENCE NUMBER \
                   --ckpt=CHECKPOINTS
```

## Localization result

| Sequence       | Localization Acc |
| -------------- | ---------------- |
| 01 Ape         |                  |
| 02 Benchvise   |                  |
| 04 Camera      | 97.49%           |
| 05 Can         | 91.39%           |
| 06 Cat         | 90.52%           |
| 10 Eggbox      | 95.99%           |
| 11 Glue        | 94.04%           |
| 13 Iron        | 95.89%           |

