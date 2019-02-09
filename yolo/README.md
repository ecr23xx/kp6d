# Object Detector

## Training

Here I use [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) to train.

### Compile

* [Compile on Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux)
* [Compile on Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows)

### Prepare data

Run following script to prepare training data for darknet.

```
$ python prepare_linemod.py --seq SEQ
```

For details you can read [How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).

### Train

Run following script to train darknet on LINEMOD.

```
$ sh train.sh
```

## Evaluation

