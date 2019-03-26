cd /home/penggao/projects/pose/kp6d
python detect.py --bs=4 \
                 --reso=416 \
                 --gpu=3 \
                 --name=linemod-single \
                 --seq=02 \
                 --ckpt=02.best.weights