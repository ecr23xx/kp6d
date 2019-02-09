cd /home/penggao/projects/kp6d/yolo/darknet
export CUDA_VISIBLE_DEVICES=1
./darknet detector train data/01/ape.data cfg/single.cfg darknet53.conv.74