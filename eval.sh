# CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --num_of_gpus=4 --config=config/cityscapes.yaml TEST.scales "[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]"
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --num_of_gpus=4 --config=config/cityscapes.yaml TEST.scales "[1.0]"
