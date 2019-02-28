# CUDA_VISIBLE_DEVICES=0,2 python train.py --nClasses=17 --seq=01 --kptype=random --optMethod=adam --trainBatch=48 --dataset=linemod --datatype=gt --loadModel=../exp/linemod/01_17_random_gt/best.pkl --addDPG
# CUDA_VISIBLE_DEVICES=2,3 python train.py --nClasses=17 --seq=01 --kptype=sift --optMethod=adam --trainBatch=48 --dataset=linemod --datatype=gt
cd /home/penggao/projects/kp6d/keypoint/train_sppe
export CUDA_VISIBLE_DEVICES=1
python train.py --nClasses=17 \
                --seq=05 \
                --kptype=sift \
                --optMethod=adam \
                --trainBatch=48 \
                --dataset=linemod \
                --datatype=gt \
                --loadModel=../exp/linemod/01_17_random_gt/model_300.pkl \
                --addDPG