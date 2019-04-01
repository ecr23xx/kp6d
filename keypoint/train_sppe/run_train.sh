cd /home/penggao/projects/pose/kp6d/keypoint/train_sppe
export CUDA_VISIBLE_DEVICES=3
python train.py --nClasses=17 \
                --dataset=linemod \
                --trainBatch=48 \
                --optMethod=adam \
                --datatype=gt \
                --seq=10 \
                --kptype=cluster \
                # --loadModel ../exp/linemod/10_17_cluster_gt/model_best.pkl \
                # --addDPG