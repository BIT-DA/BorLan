#!/usr/bin/env bash
python src/main_borlan.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --projector_dim 1024 --seed 2023 --backbone resnet50 --label_ratio 15 --pretrained --text_model_name bert
python src/main_borlan.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --projector_dim 1024 --seed 2023  --backbone resnet50 --label_ratio 30 --pretrained --text_model_name bert
python src/main_borlan.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --projector_dim 1024 --seed 2023  --backbone resnet50 --label_ratio 50 --pretrained --text_model_name bert

python src/main_borlan.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --projector_dim 1024 --seed 2023 --backbone resnet50  --label_ratio 15 --pretrained --text_model_name bert
python src/main_borlan.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --projector_dim 1024 --seed 2023 --backbone resnet50  --label_ratio 30 --pretrained --text_model_name bert
python src/main_borlan.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --projector_dim 1024 --seed 2023 --backbone resnet50  --label_ratio 50 --pretrained --text_model_name bert

python src/main_borlan.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --projector_dim 1024 --seed 2023 --backbone resnet50 --label_ratio 15 --pretrained --text_model_name bert --lr 3e-4
python src/main_borlan.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --projector_dim 1024 --seed 2023 --backbone resnet50 --label_ratio 30 --pretrained --text_model_name bert --lr 3e-4
python src/main_borlan.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --projector_dim 1024 --seed 2023 --backbone resnet50 --label_ratio 50 --pretrained --text_model_name bert --lr 3e-4

