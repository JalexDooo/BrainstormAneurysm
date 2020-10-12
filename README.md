# The Segmentation of Intracranial Saccular Aneurysm and BraTS 2020 Challenge

## V0.1
- 数据预处理和增强 -> 在输入图像上每个通道添加一个随机的强度偏移（图像std的+-0.1）和放缩（0.9-1.1）
- 在三个轴上应用概率为0.5的随机的镜像反转。
- 在一定的卷积层上应用dropout层，设置为0.2。
- 下采样使用卷积进行下采样。

## V0.2
- 输出loss与dice
- 画loss与dice图