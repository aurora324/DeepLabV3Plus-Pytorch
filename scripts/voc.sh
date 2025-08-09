nohup python -u main.py \
    --save_val_results \
    --crop_size 64 \
    --batch_size 1024 \
    --val_batch_size 1024 > "voc_train.log" 2>&1 &

nohup python -u predict.py \
    --model deeplabv3plus_mobilenet \
    --input /root/code/DeepLabV3Plus-Pytorch/samples/1_image.png \
    --gpu_id 0 \
    --crop_size 513 \
    --output_stride 16 \
    --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth \
    --save_val_results './predict' > "predict.log" 2>&1 &

ps -aux | grep main.py
ps -aux | grep main.py
