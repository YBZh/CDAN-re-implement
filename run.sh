#!/bin/bash
python main.py  --data_path_source /data/domain_adaptation/Office31/  --src amazon  --arch resnet50  --numclass_da 1  --numclass_s 31 \
                --data_path_source_t /data/domain_adaptation/Office31/ --src_t webcam  --batch_size 128 --lr 1e-2  --domain_feature random_bilinear \
                --data_path_target  /data/domain_adaptation/Office31/     --tar webcam  --gamma 3.1622776601683795  --lrplan dao --adv_loss reverse \
                --print_freq 1 --epochs 200 --schedule 40 80 120 160 200 200  --workers 8   --momentum 0.9 --weight_decay 1e-4 --log random_matrix \
                --pretrained

#python main.py  --data_path_source /data/domain_adaptation/Office31/  --src amazon  --arch resnet50  --numclass_da 1  --numclass_s 31 \
#                --data_path_source_t /data/domain_adaptation/Office31/ --src_t webcam  --batch_size 128 --lr 1e-2  --domain_feature full_bilinear \
#                --data_path_target  /data/domain_adaptation/Office31/     --tar webcam  --gamma 3.1622776601683795  --lrplan dao --adv_loss reverse \
#                --print_freq 1 --epochs 200 --schedule 40 80 120 160 200 200  --workers 8   --momentum 0.9 --weight_decay 1e-4 --log round2_resverse_gradient \
#                --pretrained   ## --pretrained_fc --pretrained_checkpoint ../baseline/only_source_final_amazon2webcam_31_resnet50/model_best.pth.tar
