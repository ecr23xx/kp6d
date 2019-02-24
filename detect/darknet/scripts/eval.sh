cd /home/penggao/projects/kp6d/detect/darknet/
export CUDA_VISIBLE_DEVICES=3
./darknet detector map data/01/ape.data cfg/single.cfg backup/01/single_1300.weights -dont_show -thresh 0.5