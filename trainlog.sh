# ---------------------------num_gpu==4-------------------------------
# task 2628
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNet' --random_epoch=100 --batch_size=12  #--load_model_path='task_Random__final_epoch.pth'
# task 2748
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='ZeroResNet' --random_epoch=100 --batch_size=12  #--load_model_path='task_Random__final_epoch.pth'
#task 2796
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='RoResNet' --random_epoch=100 --batch_size=12
#task
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNetFix1' --random_epoch=100 --batch_size=12

# 优化1
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNet_V1' --random_epoch=200 --batch_size=8

# predict
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020val_NormalResNet_random/'
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='ZeroResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020val_ZeroResNet_random/'
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='RoResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__epoch_95.pth' --predict_nibable_path='./brats2020val_RoResNet_epoch_95/'
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2018_val_moduletest6_random/' --val_root_path='/home/sunjindong/dataset/MICCAI_BraTS_2018_Data_Validation'
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNet_V1' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020val_NormalResNet_V1_random/'


# Redesigned code for training.
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='NormalResNet' --training_use_gpu_num=4 --training_batch_size=12 --model_vae_flag=False --training_max_epoch=200 --training_lr_decay=0.99
cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='NormalResNet' --training_use_gpu_num=4 --training_batch_size=8 --model_vae_flag=False --training_max_epoch=300 --training_lr_decay=0.9

# Redesigned code for validation.
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py val --model='NormalResNet' --model_vae_flag=False --predict_path='./NResNet_Validation_/' --val_load_model='epoch_200.pth'
