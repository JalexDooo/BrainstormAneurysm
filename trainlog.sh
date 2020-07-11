#cd /home/aneu/BrainstormAneurysm && python main.py Aneu_train --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=4 --lr=0.001
#cd /home/aneu/BrainstormAneurysm && python main.py Aneu_predict --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth'
#cd /home/aneu/BrainstormAneurysm && python main.py Aneu_test --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --aneu_path='/home/aneu/dataset/final_val_dataset' --load_model_path='task_Random__final_epoch.pth'

# ----------------------------------------------------------
# task lr==0.0001 ->
cd /home/aneu/BrainstormAneurysm && python main.py Aneu_train --task='Random' --model='ZeroResNet' --max_epoch=80 --batch_size=8 --lr=0.001 #--load_model_path='task_Random__epoch_33.pth'
# task lr==0.0001 ->
#cd /home/aneu/BrainstormAneurysm && python main.py Aneu_train --task='Random' --model='NormalResNet' --max_epoch=80 --batch_size=4 --lr=0.001

