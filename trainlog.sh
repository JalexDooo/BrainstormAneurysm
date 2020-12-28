# Redesigned code version 3.0

# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --description='+r0'


###################### paper experiments
# Baseline r0
# task 5442
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=1.0 --description='_r0_'

# task 5324 训练出错 model='_r0' line 9
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=1.0 --description='_+r0'

# task 
cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=0.1 --description='_+r0_dec'

