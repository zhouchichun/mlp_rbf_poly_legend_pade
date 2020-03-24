

MLP_config={
        'which':'MLP',
        'struc':[[128,'sigmoid']],#'tanh','relu'
        'var_name':'real',
        'highest':1,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }
Con_config={
        'which':'Con',
        'struc':[[8,'tanh']],#'tanh','relu'
        'var_name':'real',
        'highest':1,
        "hidden_nodes":16,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        
        }

RBF_config={
        'which':'RBF',
        'var_name':'real',
        'highest':1,
        "hidden_nodes":64,#RBF node
        "ini_name":"tru_norm",#"tru_norm","xavier","const","scal","uniform",orth
        }
        
Poly_config={
        'which':'Poly',
        'var_name':'real',
        'order':10,
        'highest':1,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }

Leg_config={
        'which':'Leg',
        'var_name':'real',
        'order':10,
        'highest':1,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }

Pade_config={
        'which':'Pade',
        'var_name':'real',
        'order_up':4,
        'order_down':5,
        'highest':1,
        "ini_name":"uniform",#"tru_norm","xavier","const","scal","uniform",orth
        }
train_config={
        'clip':False,#0.05,
        'CKPT':'ckpt',
        "BATCHSIZE":1000,
        "MAX_ITER":5000,
        'STEP_EACH_ITER':500,
        'STEP_SHOW':200,
        'EPOCH_SAVE':20,
        "LEARNING_RATE":0.00008,
        "bound_weight":1,
        "step_unbound":5,
        "decay":False,
        "test_line":False,
        "is_plot":True
}
