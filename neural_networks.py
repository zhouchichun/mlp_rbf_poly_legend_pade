# Author Zhouchichun
# 59338158@qq.com, zhouchichun@tju.edu.cn, zhouchichun@dali.edu.cn
"""
    构建一个网络,一维输入,该网络可以是MLP,RBF,Poly或者Four
    给出值，给出N阶导数，如，一阶，二阶，三阶导数等
    方法如下：
        构建
        phi = MLP(config)
        phi = RBF(config)
        phi = POLY(config)
        phi = FOUR(config)
        自变量
        feed_dict={phi.input:
                  }
        函数值
        phi.value      
        导数值
        phi.d_values[1],phi.d_values[2],phi.d_values[3]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import Legend
import para_fun as P
def get_initializer(ini_name):
    if ini_name=="tru_norm":
        weight_initialization =  tf.truncated_normal_initializer(stddev=0.1)
    elif ini_name=="xavier":
        weight_initialization =  tf.contrib.layers.xavier_initializer()
    elif ini_name=="const":
        weight_initialization = tf.constant_initializer(0.1)
    elif ini_name=="uniform":
        weight_initialization =tf.random_uniform_initializer()
    elif ini_name=="scal":
        weight_initialization = tf.variance_scaling_initializer()
    elif ini_name=="orth":
        weight_initialization = tf.orthogonal_initializer()
    else:
        print("初始化方式错误，请检查config.py文件，初始函数从['tru_norm',\
        'xavier','const','uniform','scal','orth']")
        exit()
    return weight_initialization

def get_activate(act_name):
    if act_name=="sigmoid":
        return tf.nn.sigmoid
    elif act_name=="tanh":
        return tf.nn.tanh
    elif act_name=="relu":
        return tf.nn.relu
    else:
        print("激活函数配置错误，请检查config.py文件，激活函数从['tanh','relu','sigmoid']")
        exit()

class Leg: 
    def __init__(self,config):
        self.n_input = 1
        self.order =  config['order']
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = 1
        self.highest=config["highest"]
        ######
        print('建立LEG网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()
        self.build_derivation()

#############搭建Poly指定层数，指定每一曾的神经元个数
    def test_line(self,k,b):
        grad=tf.Variable(k,dtype=tf.float64,trainable=False,name="grad")
        inter=tf.Variable([b],dtype=tf.float64,trainable=False,name="inter")
        self.line=grad*self.input+inter
        self.d_line = tf.gradients(self.line, self.input)[0]
        
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
        
    def build_bound(self):
        self.power_1 = tf.get_variable(self.var_name + 'pow_1' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.power_2 = tf.get_variable(self.var_name + 'pow_2' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.xa,self.ya=P.a,P.A
        self.xb,self.yb=P.b,P.B
        self.bound_cons=tf.pow(self.input-self.xa,self.power_1)*tf.pow(self.xb-self.input,self.power_2)
        
        self.k=(self.yb-self.ya)/(self.xb-self.xa)
        self.inter=(self.xb*self.ya-self.yb*self.xa)/(self.xb-self.xa)
        self.value=self.value_*self.bound_cons\
        +self.input*self.k+self.inter
    def build_value(self):
        print("建立网络结构")
        self.value_=0
        self.legends=Legend.give_legend(self.input,self.order)
        for i in range(1,self.order+1):
            w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                initializer=tf.constant([0.0001/(i**2)],tf.float64),
                                dtype=tf.float64)
            self.value_ +=w*self.legends[i]
        self.build_bound()
##############
    def build_derivation(self):
        print("建立导数")
        self.d_values={}
        for i in range(1,self.highest+1):
            if i==1:
                 self.d_values[i]=tf.gradients(self.value,self.input)[0]
            else:
                 self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
        
class MLP: 
    def __init__(self,config):
        self.struc =  config['struc']
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = 1
        self.n_input=1
        self.highest=config["highest"]
        ######
        print('建立MLP网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()
        self.build_derivation()

#############搭建MLP指定层数，指定每一曾的神经元个数
    def test_line(self,k,b):
        grad=tf.Variable(k,dtype=tf.float64,trainable=False,name="grad")
        inter=tf.Variable([b],dtype=tf.float64,trainable=False,name="inter")
        self.line=grad*self.input+inter
        self.d_line = tf.gradients(self.line, self.input)[0]
        
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
    def build_bound(self):
        self.power_1 = tf.get_variable(self.var_name + 'pow_1' , 
                                    initializer=tf.constant(2.0,tf.float64), )
        self.power_2 = tf.get_variable(self.var_name + 'pow_2' , 
                                    initializer=tf.constant(2.0,tf.float64), )
        self.xa,self.ya=P.a,P.A
        self.xb,self.yb=P.b,P.B
        self.bound_cons=tf.pow(self.input-self.xa,self.power_1)*tf.pow(self.xb-self.input,self.power_2)
        
        self.k=(self.yb-self.ya)/(self.xb-self.xa)
        self.inter=(self.xb*self.ya-self.yb*self.xa)/(self.xb-self.xa)
        self.value=self.value_*self.bound_cons\
        +self.input*self.k+self.inter
    def build_value(self):
        print("建立网络结构")
        for i,stru in enumerate(self.struc):
            this_num,this_act=stru
            activate=get_activate(this_act)
            if i == 0:
                w = tf.get_variable(self.var_name + 'weight_' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.input, w), b))
               
            else:
                w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.layer, w), b))
                
        w =  tf.get_variable(self.var_name+'weight_' + str(len(self.struc)), 
                            shape=[self.struc[-1][0], self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        b =tf.get_variable(self.var_name+'bias_' + str(len(self.struc)), 
                            shape=[self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        self.value_=tf.matmul(self.layer, w) + b
        #self.value_=tf.matmul(self.layer, w) + b
        #self.value_=self.value
        self.build_bound()
##############
    def build_derivation(self):
        print("建立导数")
        self.d_values={}
        for i in range(1,self.highest+1):
            if i==1:
                 self.d_values[i]=tf.gradients(self.value,self.input)[0]
            else:
                 self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
        
class RBF: 
    def __init__(self,config):
        self.n_input=1
        self.hidden_nodes=config["hidden_nodes"]
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = 1
        self.highest=config["highest"]
        #######
        print('建立RBF网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()
        self.build_derivation()

#############搭建MLP指定层数，指定每一曾的神经元个数
    def test_line(self,k,b):
        grad=tf.Variable(k,dtype=tf.float64,trainable=False,name="grad")
        inter=tf.Variable([b],dtype=tf.float64,trainable=False,name="inter")
        self.line=grad*self.input+inter
        self.d_line = tf.gradients(self.line, self.input)[0]
        
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
    
    def build_bound(self):
        self.power_1 = tf.get_variable(self.var_name + 'pow_1' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.power_2 = tf.get_variable(self.var_name + 'pow_2' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.xa,self.ya=P.a,P.A
        self.xb,self.yb=P.b,P.B
        self.bound_cons=tf.pow(self.input-self.xa,self.power_1)*tf.pow(self.xb-self.input,self.power_2)
        
        self.k=(self.yb-self.ya)/(self.xb-self.xa)
        self.inter=(self.xb*self.ya-self.yb*self.xa)/(self.xb-self.xa)
        self.value=self.value_*self.bound_cons\
        +self.input*self.k+self.inter
    def build_value(self):
        print("建立RBF网络")
        self.distance=[]
        
        self.delta = tf.get_variable(self.var_name+"_delta",
                                     shape      = [self.hidden_nodes],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float64)
        self.delta_2=self.delta**2
        for i in range(self.hidden_nodes):
            this_center = tf.get_variable(self.var_name + 'center_' + str(i),
                                     shape      = [self.n_input],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float64)
            this_dist=tf.reshape(tf.reduce_sum((self.input - this_center)**2,axis=1),[-1,1])
            self.distance.append(this_dist)
           
        self.distance_ca=tf.concat(self.distance,axis=1)
        self.distance_ca=tf.reshape(self.distance_ca,[-1,self.hidden_nodes])
        self.out_hidden=tf.exp(-1.0*(self.distance_ca/self.delta_2))
       
        self.w_h2o=tf.get_variable(self.var_name + 'w_h2o',
                                   shape       = [self.hidden_nodes,self.n_output],
                                   initializer = self.weight_initialization,
                                   dtype       = tf.float64)
        self.bias=tf.get_variable(self.var_name+"_bias",
                                shape=[self.n_output],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
        self.value_=tf.matmul(self.out_hidden,self.w_h2o)+self.bias
        self.build_bound()

##############
    def build_derivation(self):
        print("建立导数")
        self.d_values={}
        for i in range(1,self.highest+1):
            if i==1:
                 self.d_values[i]=tf.gradients(self.value,self.input)[0]
            else:
                 self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
       

class Poly: 
    def __init__(self,config):
        self.n_input = 1
        self.order =  config['order']
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = 1
        self.highest=config["highest"]
        ######
        print('建立Poly网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()
        self.build_derivation()

#############搭建Poly指定层数，指定每一曾的神经元个数
    def test_line(self,k,b):
        grad=tf.Variable(k,dtype=tf.float64,trainable=False,name="grad")
        inter=tf.Variable([b],dtype=tf.float64,trainable=False,name="inter")
        self.line=grad*self.input+inter
        self.d_line = tf.gradients(self.line, self.input)[0]
        
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
         
    def build_bound(self):
        self.power_1 = 2.0#tf.get_variable(self.var_name + 'pow_1' , 
                                    #initializer=tf.constant(1.0,tf.float64), )
        self.power_2 = 2.0#tf.get_variable(self.var_name + 'pow_2' , 
                                    #initializer=tf.constant(1.0,tf.float64), )
        self.xa,self.ya=P.a,P.A
        self.xb,self.yb=P.b,P.B
        self.bound_cons=0.01*tf.pow(self.input-self.xa,self.power_1)*tf.pow(self.xb-self.input,self.power_2)
        
        self.k=(self.yb-self.ya)/(self.xb-self.xa)
        self.inter=(self.xb*self.ya-self.yb*self.xa)/(self.xb-self.xa)
        self.value=self.value_*self.bound_cons\
        +self.input*self.k+self.inter
    def build_value(self):
        print("建立网络结构")
        order=self.order
        self.value_=0
       
        self.w={}
        for i in range(1,order+1):
            w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                initializer=tf.constant([0.0001/(i**2)],tf.float64),
                                dtype=tf.float64)
            self.w[i]={}
            for j in range(i+1):
                if j==0:
                    self.w[i][j]=w
                else:
                    coe=[x for x in range(1,i+1)][-j:]
                    ret=1
                    for c in coe:
                        ret *=c
                    self.w[i][j]=ret*w
        for ordd,value in self.w.items():
            tmp=1
           
            for _ in range(ordd):
                tmp *=self.input
               
 
            self.value_ +=tmp * value[0]
            
            
        self.b = tf.get_variable(self.var_name + 'bias', 
                                    initializer=tf.constant([0.5],dtype=tf.float64), 
                                    dtype=tf.float64)
        self.value_ +=self.b
        self.build_bound()
##############
    def build_derivation(self):
        print("建立导数")
        self.d_values={}
        for i in range(1,self.highest+1):
            if i==1:
                 self.d_values[i]=tf.gradients(self.value,self.input)[0]
            else:
                 self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
           


class Con: 
    def __init__(self,config):
        self.n_input=1
        self.struc =  config['struc']
        self.var_name = config['var_name']
        self.hidden_nodes=config["hidden_nodes"]
       
        self.ini_name =config["ini_name"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = 1
        self.highest=config["highest"]
        #######
        print('建立RBF网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()
        self.build_derivation()

#############搭建con指定层数，指定每一曾的神经元个数
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
         

    def build_value_MLP(self):
        print("建立MLP网络结构")
        for i,stru in enumerate(self.struc):
            this_num,this_act=stru
            activate=get_activate(this_act)
            if i == 0:
                w = tf.get_variable(self.var_name + 'weight_' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.input, w), b))
                
            else:
                w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.layer, w), b))
             
        w =  tf.get_variable(self.var_name+'weight_' + str(len(self.struc)), 
                            shape=[self.struc[-1][0], self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        b =tf.get_variable(self.var_name+'bias_' + str(len(self.struc)), 
                            shape=[self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        self.value_mlp=tf.matmul(self.layer, w) + b
        


    def build_value_RBF(self):
        print("建立RBF网络")
        self.distance=[]
       
        self.delta = tf.get_variable(self.var_name+"_delta",
                                     shape      = [self.hidden_nodes],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float64)
        self.delta_2=self.delta**2
        for i in range(self.hidden_nodes):
            this_center = tf.get_variable(self.var_name + 'center_' + str(i),
                                     shape      = [self.n_input],
                                     initializer= self.weight_initialization,
                                     dtype      = tf.float64)
            this_dist=tf.reshape(tf.reduce_sum((self.input - this_center)**2,axis=1),[-1,1])
            self.distance.append(this_dist)
            
        self.distance_ca=tf.concat(self.distance,axis=1)
        self.distance_ca=tf.reshape(self.distance_ca,[-1,self.hidden_nodes])
        self.out_hidden=tf.exp(-1.0*(self.distance_ca/self.delta_2))
        
        self.w_h2o=tf.get_variable(self.var_name + 'w_h2o',
                                   shape       = [self.hidden_nodes,self.n_output],
                                   initializer = self.weight_initialization,
                                   dtype       = tf.float64)
        self.bias=tf.get_variable(self.var_name+"_bias",
                                shape=[self.n_output],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
        self.value_rbf=tf.matmul(self.out_hidden,self.w_h2o)+self.bias
    
    def build_bound(self):
        self.power_1 = tf.get_variable(self.var_name + 'pow_1' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.power_2 = tf.get_variable(self.var_name + 'pow_2' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.xa,self.ya=P.a,P.A
        self.xb,self.yb=P.b,P.B
        self.k=(self.yb-self.ya)/(self.xb-self.xa)
        self.inter=(self.xb*self.ya-self.yb*self.xa)/(self.xb-self.xa)
        self.value=self.value*(self.input-self.xa)**self.power_1*(self.xb-self.input)**self.power_2\
        +self.input*self.k+self.inter
    def build_value(self):
        self.build_value_MLP()
        self.build_value_RBF()
        self.mlp_rbf=tf.get_variable(self.var_name+"_mlp_weight",
                                shape=[2,1],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
        self.mlp_rbf=tf.nn.softmax(self.mlp_rbf)
        self.value_concat=tf.concat([self.value_rbf,self.value_mlp],1)
        
        self.value=tf.matmul(self.value_concat,self.mlp_rbf)#self.value_rbf+self.value_mlp
        self.build_bound()
##############
    def build_derivation(self):
        print("建立导数")
        self.d_values={}
        for i in range(1,self.highest+1):
            if i==1:
                 self.d_values[i]=tf.gradients(self.value,self.input)[0]
            else:
                 self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
        

class Pade: 
    def __init__(self,config):
        self.var_name = config['var_name']
        self.ini_name =config["ini_name"]
        self.order_up=config["order_up"]
        self.order_down=config["order_down"]
        self.weight_initialization=get_initializer(self.ini_name)
        self.n_output = 1
        self.n_input=1
        self.highest=config["highest"]
        ######
        print('建立MLP网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######
        self.construct_input()
        self.build_value()
        self.build_derivation()

#############搭建MLP指定层数，指定每一曾的神经元个数
    def test_line(self,k,b):
        grad=tf.Variable(k,dtype=tf.float64,trainable=False,name="grad")
        inter=tf.Variable([b],dtype=tf.float64,trainable=False,name="inter")
        self.line=grad*self.input+inter
        self.d_line = tf.gradients(self.line, self.input)[0]
        
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])   
         
    def build_bound(self):
        self.power_1 = tf.get_variable(self.var_name + 'pow_1' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.power_2 = tf.get_variable(self.var_name + 'pow_2' , 
                                    initializer=tf.constant(1.0,tf.float64), )
        self.xa,self.ya=P.a,P.A
        self.xb,self.yb=P.b,P.B
        self.k=(self.yb-self.ya)/(self.xb-self.xa)
        self.inter=(self.xb*self.ya-self.yb*self.xa)/(self.xb-self.xa)
        self.value=self.value*(self.input-self.xa)**self.power_1*(self.xb-self.input)**self.power_2\
        +self.input*self.k+self.inter
    def build_value(self):
        print("建立网络结构")
        self.value_up=0
        self.value_down=0
        for i in range(1,self.order_up+1):
            w = tf.get_variable(self.var_name + 'weight_up' + str(i), 
                                initializer=tf.constant([0.01/i],tf.float64),
                                dtype=tf.float64)
            tmp=w*tf.pow(self.input,i)
            self.value_up +=tmp
        for i in range(1,self.order_down+1):
            w = tf.get_variable(self.var_name + 'weight_down' + str(i), 
                                initializer=tf.constant([0.01/i],tf.float64),
                                dtype=tf.float64)
            tmp=w*tf.pow(self.input,i)
            self.value_down +=tmp
        self.value=tf.divide(self.value_up,self.value_down)
        self.build_bound()
##############
    def build_derivation(self):
        print("建立导数")
        self.d_values={}
        for i in range(1,self.highest+1):
            if i==1:
                 self.d_values[i]=tf.gradients(self.value,self.input)[0]
            else:
                 self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
        