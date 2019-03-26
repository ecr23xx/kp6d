cd /home/penggao/projects/pose/kp6d/detect/darknet/
export CUDA_VISIBLE_DEVICES=3
./darknet detector map data/05/can.data cfg/single.cfg backup/05.best.weights -dont_show -thresh 0.5