cd /home/penggao/projects/kp6d/detect/darknet
export CUDA_VISIBLE_DEVICES=0
./darknet detector train data/01/ape.data cfg/single.cfg darknet53.conv.74 -map