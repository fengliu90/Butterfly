# Butterfly for Wildly Unsupervised Domain Adaptation
This is the official site for the paper "Butterfly: One-step Approach towards Wildly Unsupervised Domain Adaptation" (http://128.84.4.34/pdf/1905.07720).

# Data download
You can download MNIST and SYND data used in our paper from https://drive.google.com/file/d/1f4sepb0JXeSSftfAWd6KPa_AVoWA28rj/view?usp=sharing. You need to extract it to ./data file.

# Butterfly Runing
There are eight tasks in Butterfly_total.py file:
1. M2S P20
2. M2S P45
3. M2S S20
4. M2S S45
5. S2M P20
6. S2M P45
7. S2M S20
8. S2M S45

You can specify one task and run Butterfly using 

python Butterfly_total.py --Task_type 1 --> get results for the first task.
