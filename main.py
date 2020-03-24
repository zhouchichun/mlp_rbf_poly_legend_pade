import tensorflow as tf 

import numpy as np 
import neural_networks as N
import matplotlib.pyplot as plt
import sys, getopt
from utils import give_batch
import config as C
import time    
import para_fun as P
import logging


class the_net():
    def __init__(self,train_config,stru_config,loggin):
        self.net_name=stru_config["which"]
        self.loggin=loggin
        self.loggin.info("now initialize the net with para:")
        for item,value in train_config.items():
            self.loggin.info(str(item))
            self.loggin.info(str(value))
            self.loggin.info("-----------------------------")
        
        self.save_path=train_config["CKPT"]+"_"+self.net_name
        self.clip=train_config["clip"]
        self.learning_rate=train_config["LEARNING_RATE"]
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size = train_config["BATCHSIZE"]
        self.max_iter = train_config["MAX_ITER"]
        #self.step_unbound=train_config["step_unbound"]
        self.epoch_save=train_config["EPOCH_SAVE"]
        self.step_each_iter=train_config['STEP_EACH_ITER']
        self.step_show=train_config['STEP_SHOW']
        self.global_steps = tf.Variable(0, trainable=False)  
        self.stru_config=stru_config
        self.decay=train_config["decay"]#False
        self.test_line=train_config["test_line"]#        
        #self.bound_weight=train_config["bound_weight"]
        self.is_plot=train_config["is_plot"]
        
###############################
#P.para
##############################
        self.sess=tf.Session()
        print("openning sess")
        self.loggin.info("openning sess")
        self.build_net()
        print("building net")
        self.loggin.info("building net")
        self.build_opt()
        print("building opt")
        self.loggin.info("building opt")
        self.saver=tf.train.Saver(max_to_keep=3)
        self.initialize()
        print("net initializing")
        self.loggin.info("net initializing")
        self.D=give_batch([P.a,P.b])
    
    def build_net(self):
        if self.net_name=="MLP":
            self.y = N.MLP(self.stru_config)
        elif self.net_name=="RBF":
            self.y = N.RBF(self.stru_config)
        elif self.net_name=="Poly":
            self.y = N.Poly(self.stru_config)
        elif self.net_name=="Con":
            self.y = N.Con(self.stru_config)
        elif self.net_name=="Pade":
            self.y = N.Pade(self.stru_config)
        elif self.net_name=="Leg":
            self.y = N.Leg(self.stru_config)
        else:
            print("网络模型错误！")
            exit()
###########################################
        self.target=tf.placeholder(tf.float64, [None, 1])  
        self.loss= (P.b-P.a)*tf.reduce_mean((self.target-self.y.value)**2)
       # self.loss= 2*P.g*self.y.value

###########################################
        if self.test_line:
             print("正在做直线测试")
             grad=2.0/3.141592653
             inter=0.0
             self.y.test_line(grad,inter)
             self.loss_line= (P.b-P.a)*tf.reduce_mean((1.0+self.y.d_line[0]**2)**0.5\
                                          *(2*10.0*self.y.line)**(-0.5))
             self.sess.run(tf.global_variables_initializer())
             xx=[[float(x)/self.batch_size] for x in range(1,int(3.141592653*self.batch_size))]
             
             loss_line,line,gr=self.sess.run([self.loss_line,self.y.line,self.y.d_line[0]],
                                             feed_dict={self.y.input:xx})
             print("直线的泛函值为 %s"%loss_line)
             ###########################################
             xx_rand=np.random.uniform(P.a, P.b, len(xx))
             xx_rand=np.reshape(xx_rand,[len(xx),1])
             loss_line,line,gr=self.sess.run([self.loss_line,self.y.line,self.y.d_line[0]],
                                             feed_dict={self.y.input:xx_rand})
             print("蒙特卡洛直线的泛函值为 %s"%loss_line)
             exit()

    def build_opt(self):
        if self.decay:
            self.learning_rate_d = tf.train.exponential_decay(self.learning_rate,
                                           global_step=self.global_steps,
                                           decay_steps=self.step_each_iter,decay_rate=0.95) 
        else:
            self.learning_rate_d =self.learning_rate
        
        if self.clip:
            print("导数有约束")
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_d) 
            self.gradients = self.optimizer.compute_gradients(self.loss)
            self.capped_gradients = [(tf.clip_by_value(grad, -1*self.clip, self.clip), var) \
            for grad, var in self.gradients if grad is not None]
            self.opt = self.optimizer.apply_gradients(self.capped_gradients,self.global_steps)
            
        else:
            self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate_d).minimize(self.loss,global_step=self.global_steps)

    def initialize(self):
        ckpt=tf.train.latest_checkpoint(self.save_path)
        if ckpt!=None:
            self.saver.restore(self.sess,ckpt)
            print("init from ckpt ")
            self.loggin.info("init from ckpt ")
        else:
            self.sess.run(tf.global_variables_initializer())
    def plot(self,value,real,le1,le2):
        
        plt.plot(value)
        plt.plot(real)
        
        plt.legend([le1,le2])
        plt.show()
    def train(self):
        st=time.time()
        for epoch in range(self.max_iter):
            print("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            self.loggin.info("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            for step in range(self.step_each_iter):
                intx=self.D.inter(self.batch_size)
                target=P.give_target(intx)
                #value_,bound_cons,value,loss=self.sess.run([self.y.value_,self.y.bound_cons,\
                #self.y.value,self.loss],feed_dict={self.y.input:intx})
                                        
                #plt.plot(value_)
                #plt.plot(bound_cons)
                #plt.plot(value)
                #print("=======%s"%loss)
                #plt.legend(["value_","bound","value"])
                #plt.show()
                #continue
                #exit()
                if self.decay:
                    loss,_,gs,lrd=self.sess.run([self.loss,self.opt,self.global_steps,self.learning_rate_d],\
                                        feed_dict={self.y.input:intx,self.target:target})
                    if np.isnan(loss):
                        print("梯度爆了,在%s步的时候，请调整参数初始化或者调小学习步长"%step)
                        self.loggin.error("梯度爆了,在%s步的时候，请调整参数初始化或者调小学习步长"%step)
                        exit()
                else:
                    loss,_,gs=self.sess.run([self.loss,self.opt,self.global_steps],\
                                        feed_dict={self.y.input:intx,self.target:target})
                    lrd=self.learning_rate_d
                    if np.isnan(loss):
                        print("梯度爆了,在%s步的时候，请调整参数初始化或者调小学习步长"%step)
                        self.loggin.error("梯度爆了,在%s步的时候"%step)
                        loss,value=self.sess.run([self.y.value,self.loss],\
                                        feed_dict={self.y.input:intx,self.target:target})
                        #,value_,d_value=self.sess.run([self.y.value,self.y.value_,self.y.d_values[1]],\
                        #                feed_dict={self.y.input:intx})
                        print(loss)
                        print("===============")
                        plt.plot(value)
                        plt.show()
                        #print(value_)
                        #print(d_value)
                        exit()
                if (step+1)%self.step_show==0:
                    print("loss %s, in epoch %s, in step %s, in global step %s, learning rate is %s, taks %s seconds"%(loss,epoch,step,gs,lrd,time.time()-st))
                    self.loggin.error("loss %s, in epoch %s, in step %s, in global step %s, learning rate is %s, taks %s seconds"%(loss,epoch,step,gs,lrd,time.time()-st))
                    st=time.time()
        
            if (epoch+1)%self.epoch_save==0:
                self.saver.save(self.sess, self.save_path+"/check.ckpt")
                int_x=self.D.inter(1000)
                target=P.give_target(intx)
                if self.decay:
                    value,lrd,loss=self.sess.run([self.y.value,self.learning_rate_d,self.loss], \
                                            feed_dict={self.y.input:int_x,self.target:target})
                else:
                    if self.net_name=="Con":
                        value,loss,value_mlp,value_rbf=self.sess.run([self.y.value,self.loss,self.y.value_mlp,self.y.value_rbf], \
                                            feed_dict={self.y.input:int_x,self.target:target}) 
                    else:
                        value,loss=self.sess.run([self.y.value,self.loss], \
                                            feed_dict={self.y.input:int_x,self.target:target})
                    lrd=self.learning_rate_d
                if self.is_plot:
                    yy=P.give_target(int_x)
                    real=np.reshape(yy,[len(yy)])
                    print("散点一共有%s 个"%len(int_x))
                    if self.net_name=="Con":
                        self.plot(value_mlp,real,"MLP","analytic result")
                        self.plot(value_rbf,real,"RBF","analytic result")
                    else:
                        self.plot(value,real,self.net_name+"'s result -- %s"%loss,"analytic result --%s"%P.exact)
                    
                print("Model saved in path: %s in epoch %s. learning_rate is %s, loss is %s" % (self.save_path,epoch,lrd,loss))
                self.loggin.info("Model saved in path: %s in epoch %s. learning_rate is %s, loss is %s" % (self.save_path,epoch,lrd,loss))
            

if __name__ == '__main__':
    which=sys.argv[1]
    if which=="MLP":
        logger = logging.getLogger(C.MLP_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.MLP_config["which"]+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.MLP_config,logger)
        main_net.train()
    if which=="Poly":
        logger = logging.getLogger(C.Poly_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.Poly_config["which"]+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.Poly_config,logger)
        main_net.train()

    if which=="RBF":
        logger = logging.getLogger(C.RBF_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.RBF_config["which"]+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.RBF_config,logger)
        main_net.train()
    if which=="Con":
        logger = logging.getLogger(C.Con_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.Con_config["which"]+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.Con_config,logger)
        main_net.train()
    if which=="Pade":
        logger = logging.getLogger(C.Pade_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.Pade_config["which"]+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.Pade_config,logger)
        main_net.train()
    if which=="Leg":
        logger = logging.getLogger(C.Leg_config["which"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(C.Leg_config["which"]+"_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        main_net=the_net(C.train_config,C.Leg_config,logger)
        main_net.train()
