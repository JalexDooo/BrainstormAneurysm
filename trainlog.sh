# ---------------------------num_gpu==4-------------------------------
# task 3473
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNet' --random_epoch=100 --batch_size=32  #--load_model_path='task_Random__final_epoch.pth'
# task 3518
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='ZeroResNet' --random_epoch=100 --batch_size=32  #--load_model_path='task_Random__final_epoch.pth'
#task 3527
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='RoResNet' --random_epoch=100 --batch_size=32
#task
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNetFix1' --random_epoch=100 --batch_size=32

<<<<<<< HEAD
# predict
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020val_NormalResNet_random/'
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='ZeroResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020val_ZeroResNet_random/'
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='RoResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__epoch_95.pth' --predict_nibable_path='./brats2020val_RoResNet_epoch_95/'
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2018_val_moduletest6_random/' --val_root_path='/home/sunjindong/dataset/MICCAI_BraTS_2018_Data_Validation'
=======
# ----------------------------------------------------------
# task 3483 lr==0.001 && task lr==0.0001
cd /home/aneu/BrainstormAneurysm && python main.py Aneu_train --task='Random' --model='ZeroResNet' --max_epoch=80 --batch_size=8 --lr=0.001
cd /home/aneu/BrainstormAneurysm && python main.py Aneu_train --task='Random' --model='ZeroResNet' --max_epoch=20 --batch_size=8 --lr=0.0001 --load_model_path='task_Random__final_epoch.pth' --aneu_path='/home/aneu/dataset/dongmai_dark'
# task lr==0.0001 ->
#cd /home/aneu/BrainstormAneurysm && python main.py Aneu_train --task='Random' --model='NormalResNet' --max_epoch=80 --batch_size=4 --lr=0.001
>>>>>>> 04555ae19b35fd22d3a3d0df2711d90c6248074d

# 第二轮
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNet' --random_epoch=100 --batch_size=8 --lr=0.0001 --load_model_path='task_Random__final_epoch.pth'
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020val_NormalResNet_random/'

# task 3883 修改后 4000
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNetFix1' --random_epoch=100 --batch_size=8
#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNetFix1' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020val_NormalResNetFix1_random/'

#cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_train_random --task='Random' --model='NormalResNet' --random_epoch=150 --batch_size=8 --lr=0.0005 --load_model_path='task_Random__final_epoch.pth'
cd /home/sunjindong/BrainstormBraTS2020 && python main.py multi_val_random --task='Random' --model='NormalResNet' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth' --predict_nibable_path='./brats2020test_NormalResNet_random/' --val_root_path='/home/sunjindong/dataset/BraTS_TestData_JSun_paper31/MICCAI_BraTS2020_TestingData'
