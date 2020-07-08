#cd /home/aneu/Brainstorm && python main.py Aneu_train --task='Random' --model='ModuleTest6' --max_epoch=11 --batch_size=4 --lr=0.0001 --load_model_path='task_Random__epoch_14.pth'

#cd /home/aneu/Brainstorm && python main.py Aneu_predict --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --load_model_path='task_Random__final_epoch.pth'

#cd /home/aneu/Brainstorm && python main.py Aneu_test --task='Random' --model='ModuleTest6' --max_epoch=1 --batch_size=1 --aneu_path='/home/aneu/dataset/final_val_dataset' --load_model_path='task_Random__final_epoch.pth'
