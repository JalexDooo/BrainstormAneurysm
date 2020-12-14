# Redesigned code version 3.0

# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --description='+r0'


###################### paper experiments
# Baseline r0
# task
cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=1.0 --description='_r0'

# task 
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=1.0 --description='_+r0'

