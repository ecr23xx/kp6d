cd /home/penggao/projects/pose/kp6d/keypoint/train_sppe
export CUDA_VISIBLE_DEVICES=1

python train.py --nClasses=17 \
                --dataset=linemod \
                --trainBatch=48 \
                --optMethod=adam \
                --datatype=gt \
                --seq=10 \
                --kptype=sift \

python train.py --nClasses=17 \
                --dataset=linemod \
                --trainBatch=48 \
                --optMethod=adam \
                --datatype=gt \
                --seq=10 \
                --kptype=sift \
                --loadModel ../exp/linemod/10_17_sift_gt/model_best.pkl \
                --addDPG

cp ../exp/linemod/10_17_sift_gt_dpg/model_best.pkl ../exp/final_model-linemod-single-17-sift/10.pkl

python train.py --nClasses=17 \
                --dataset=linemod \
                --trainBatch=48 \
                --optMethod=adam \
                --datatype=gt \
                --seq=11 \
                --kptype=sift \

python train.py --nClasses=17 \
                --dataset=linemod \
                --trainBatch=48 \
                --optMethod=adam \
                --datatype=gt \
                --seq=11 \
                --kptype=sift \
                --loadModel ../exp/linemod/11_17_sift_gt/model_best.pkl \
                --addDPG

cp ../exp/linemod/11_17_sift_gt_dpg/model_best.pkl ../exp/final_model-linemod-single-17-sift/11.pkl
