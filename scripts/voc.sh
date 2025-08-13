nohup python -u main.py \
    --save_val_results \
    --crop_size 512 \
    --val_interval 100 \
    --batch_size 32 \
    --val_batch_size 32 > "main.log" 2>&1 &

nohup python -u train_m2nist.py \
    --save_val_results \
    --crop_size 64 \
    --batch_size 1024 \
    --val_batch_size 1024 > "train_m2nist.log" 2>&1 &

nohup python -u train_cam.py \
    --save_val_results \
    --crop_size 512 \
    --val_interval 1000 \
    --batch_size 32 \
    --crop_val \
    --val_batch_size 32 > "train_cam.log" 2>&1 &

nohup python -u train_cam_ot.py \
    --ckpt checkpoints/best_deeplabv3plus_resnet101_voc_os16_main_cam.pth \
    --save_val_results \
    --crop_size 512 \
    --val_interval 100 \
    --batch_size 32 \
    --crop_val \
    --val_batch_size 4 > "train_cam_ot.log" 2>&1 &

nohup python -u train_m2nist.py \
    --save_val_results \
    --crop_size 64 \
    --batch_size 1024 \
    --val_batch_size 1024 > "train_m2nist.log" 2>&1 &

nohup python -u predict.py \
    --model deeplabv3plus_mobilenet \
    --input /root/code/DeepLabV3Plus-Pytorch/samples/1_image.png \
    --gpu_id 0 \
    --crop_size 513 \
    --output_stride 16 \
    --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth \
    --save_val_results './predict' > "predict.log" 2>&1 &

python make_label.py \
  --data_root /root/autodl-tmp/ \
  --dataset voc --year '2012_aug' \
  --ckpt checkpoints/best_deeplabv3plus_resnet101_voc_os16_main_cam.pth \
  --save_dir ./voc \
  --save_val_results \
  --save_cam --cam_root /root/autodl-tmp/VOC/VOC2012/cams

ps -aux | grep main.py
ps -aux | grep train_m2nist.py
ps -aux | grep train_cls.py
ps -aux | grep train_cam.py
ps -aux | grep train_cam_ot.py
