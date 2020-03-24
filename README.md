# mlp_rbf_poly_legend_pade
使用MLP,RBF,幂次多项式，勒让德多项式以及帕得变换 拟合、逼近目标函数
为了研究不同网络的效率和效果，不失一般性的重点支持1维输入，1维输出。
# 环境搭建
- python3
- tensorflow 1.14以上

# 使用
- git clone  该项目

- 修改目标函数。 para_fun.py

  - 修改端点
  ```
  a=-1*pi/2 #起始点横坐标
  A=0       #起始点纵坐标

  b=pi/2    # 终止点横坐标
  B=2.0     #终止点纵坐标
  ```
  - 拟合任务中修改目标函数
  ```
  def give_target(x):
    x=np.reshape(x,[len(x)])
    y=np.sin(2*x**2)*np.exp(-(x-1)**2)+x*(2.7/pi)+1.35#任意修改，请保证目标函数通过上述端点。
    y=np.reshape(y,[-1,1])
    return y
  ```
  

- 配置 config.py文件
  - 设置网络结构，如MLP的层数，单层节点数目，激活函数等
  - 设置训练参数，如 batch size，step to show， is plot等
  

- 运行主文件  python3 main.py  MLP 
             python3 main.py  RBF
             python3 main.py  Poly
             python3 main.py  Leg
             python3 main.py  Pade
             python3 main.py  Com
             
- 自定义任务。配置main.py函数中 build_net 函数,
```
self.target=tf.placeholder(tf.float64, [None, 1])   #通过自己建立placeholder的方式引入 系数，如非线性项
self.loss= (P.b-P.a)*tf.reduce_mean((self.target-self.y.value)**2)#自定义loss函数，自定义任务

```
# 注意
- log文件
- ckpt保存，如果修改了模型结构，需要清除对应的ckpt文件下的模型参数文件。
