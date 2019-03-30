cd /home/penggao/projects/pose/kp6d/keypoint/train_sppe
export CUDA_VISIBLE_DEVICES=1
python train.py --nClasses=17 \
                --seq=02 \
                --kptype=sift \
                --dataset=linemod \
                --trainBatch=48 \
                --optMethod=adam \
                --datatype=gt \
                --loadModel ../exp/linemod/02_17_sift_gt/model_best.pkl \
                --addDPG