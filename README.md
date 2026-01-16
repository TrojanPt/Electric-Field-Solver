# Electric-Field-Solver

利用PyTorch神经网络内嵌泊松方程 $$\nabla^2\varphi = -\frac{\rho}{\varepsilon_0}$$ 实现求解三维静电场问题。

> [!note]
> 这只是一个练习。
> 
> 由于该模型需要针对用户输入进行训练，因此并没有任何实际用处。

### **Sample1**

**Electric Field Distribution**

<img src="image\07\electric_field_results.png" width=500/>

**Training Loss History**

<img src="image\07\training_loss_history.png" width=500/>

### **Sample2**

**Electric Field Distribution**

<img src="image\05\electric_field_distribution.png" width=500/>

**Training Loss History**

<img src="image\05\training_loss_history.png" width=500/>

### **requirements**
> torch==2.6.0+cu126
>
> numpy==2.1.2
>
> matplotlib==3.10.1
>
> *其它版本注意兼容性*
