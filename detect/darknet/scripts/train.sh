cd /home/penggao/projects/pose/kp6d/detect/darknet
export CUDA_VISIBLE_DEVICES=0
./darknet detector train data/15/phone.data cfg/single.cfg darknet53.conv.74 -map
