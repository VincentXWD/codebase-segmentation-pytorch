# CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py 
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config=config/cityscapes.yaml TEST.scales "[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]"
