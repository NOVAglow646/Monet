bash rsync -avzh --progress \
      --exclude='checkpoints/' \
      --exclude='tensorboard_log/' \
      --exclude='wandb/' \
      qxwang@222.29.2.184:/data1/qxwang/codes/EasyR1  /mnt/e/codes2