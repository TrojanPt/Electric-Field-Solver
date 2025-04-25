# Electric-Field-Solver

利用PyTorch神经网络内嵌拉普拉斯方程实现PINN(Physics-Informed Neural Networks)求解三维静电场问题。

> [!note]
> 这只是一个练习。
> 由于该模型需要针对用户输入进行训练，因此并没有任何实际用处。

### **样例输入**

**Input**

<img src="image\02\Input.png" width=500/>

### **样例输出**

**Conductors Distribution**

<img src="https://github.com/TrojanPt/Electric-Field-Solver/blob/901efc645fbb1d981ccf367fc38d66c81935a6b4/image/02/Conductors%20Distribution.png" width=500/>

**output**

<img src="image\02\Output.png" width=500/>

**Training Loss History**

<img src="image\02\Training Loss History.png" width=500/>

**Electric Field**

<img src="image\02\Electric Field.png" width=500/>

### **requirements**
> torch==2.6.0+cu126
>
> numpy==2.1.2
>
> matplotlib==3.10.1
>
> *其它版本注意兼容性*
