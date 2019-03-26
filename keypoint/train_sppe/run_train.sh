cd /home/penggao/projects/pose/kp6d/keypoint/train_sppe
export CUDA_VISIBLE_DEVICES=1,3
python train.py --nClasses=17 \
                --seq=15 \
                --kptype=sift \
                --dataset=linemod \
                --trainBatch=48 \
                --optMethod=adam \
                --datatype=gt \
                # --loadModel ../exp/linemod/15_17_sift_gt/model_460.pkl \
                # --addDPG