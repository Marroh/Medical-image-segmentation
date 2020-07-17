***该项目现只实现了基本功能，现在可以显示DCM格式的图像序列，进行脑干的分割并显示。在多类器官分割和GUI交互方面还未完善***

#### 一些说明

> Cityscape数据集在预处理时被分成了20类（把train ID=255）的类也算进去了，这是一个失误，本来应该把train ID=255和train ID=-1的一起当成背景。这个错误出在utils/dataset.py的函数IdTrans2TrainID（）中，有时间我会更改后重新上传。

#### 实验结果

![image-20200717002807423](C:\Users\mamama9503\AppData\Roaming\Typora\typora-user-images\image-20200717002807423.png)

![image-20200717002822064](C:\Users\mamama9503\AppData\Roaming\Typora\typora-user-images\image-20200717002822064.png)

<table border="0">
  <tr>    
    <td width="50%">
      <img src="C:\Users\mamama9503\AppData\Roaming\Typora\typora-user-images\image-20200717002831777.png" width="75%">
    </td>
    <td width="50%">
      <img src="C:\Users\mamama9503\AppData\Roaming\Typora\typora-user-images\image-20200717002835122.png" width="75%">
    </td>
  </tr>
</table>

![image-20200717002841257](C:\Users\mamama9503\AppData\Roaming\Typora\typora-user-images\image-20200717002841257.png)