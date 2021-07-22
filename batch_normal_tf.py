#batch normalization with dropout in tensorflow
#the imports
from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import init_weights
from sklearn.utils import shuffle

#the model
class ANN:
    def __init__(self,hidden_layer_sizes,keep_probs):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.keep_probs = keep_probs
        #list of all parameters except first and final layer
        self.all_params = []
        m1 = self.hidden_layer_sizes[0]
        self.count = 1
        for m2 in self.hidden_layer_sizes[1:]:
            #dont add bias due to batch_normalization
            w_init,_ = init_weights(m1,m2)
            W = tf.Variable(w_init,name = 'W'+str(self.count))
            #batch normalization parameters
            gamma = tf.Variable(np.ones(m2,dtype = np.float32),name = 'Gamma'+str(self.count))
            beta = tf.Variable(np.zeros(m2,dtype = np.float32),name = 'Beta'+str(self.count))
            running_mean = tf.Variable(np.zeros(m2,dtype = np.float32),trainable=False,name ='Rn_mean'+str(self.count))
            running_var = tf.Variable(np.zeros(m2,dtype = np.float32),trainable=False,name ='Rn_var'+str(self.count))
            
            self.all_params += [{'W':W,'gamma':gamma,'beta':beta,'rn_mean':running_mean,'rn_var':running_var}]
            self.count += 1
            m1 = m2
            
        
    
    #training function
    def fit(self,X,Y,batch_size,epochs,valid_size=1000,lr=0.0001,mu=0.9,decay=0.9,eps=1e-8, print_period=50,show_fig=True):
        
        lr = np.float32(lr)
        decay = np.float32(decay)
        mu = np.float32(mu)
        eps = np.float32(eps)
        
        X,Y = shuffle(X,Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        X_train,X_valid = X[:-valid_size,:],X[-valid_size:,:]
        Y_train,Y_valid = Y[:-valid_size,:],Y[-valid_size:,:]
        
        #add first and last layers
        n,m = X_train.shape
        p = Y_train.shape[1]
        
        #first layer same as every other layer
        w0_init,_ = init_weights(m,self.hidden_layer_sizes[0])
        W0 = tf.Variable(w0_init,name = 'W'+str(0))
        gamma0 = tf.Variable(np.ones(self.hidden_layer_sizes[0],dtype = np.float32),name='Gamma'+str(0))
        beta0 = tf.Variable(np.zeros(self.hidden_layer_sizes[0],dtype = np.float32),name='Beta'+str(0))
        rn_mean0 = tf.Variable(np.zeros(self.hidden_layer_sizes[0],dtype=np.float32),                   trainable=False,name='Rn_mean'+str(0))
        rn_var0 = tf.Variable(np.zeros(self.hidden_layer_sizes[0],dtype=np.float32),trainable=False,name='Rn_var'+str(0))
        self.all_params = [{'W':W0,'gamma':gamma0,'beta':beta0,'rn_mean':rn_mean0,'rn_var':rn_var0}] + self.all_params
        
        #final layer , we dont do batch normalization here
        wf_init,bf_init = init_weights(self.hidden_layer_sizes[-1],p)
        self.W = tf.Variable(wf_init,name='W'+str(self.count))
        self.b = tf.Variable(bf_init,name='b'+str(self.count))
        
        #create placeholders for data
        tfX = tf.placeholder(tf.float32,shape=(None,m),name = 'X')
        tfY = tf.placeholder(tf.float32,shape=(None,p),name = 'Y')
        
        #forward propagation during training
        Y_hat = self.forward_train(tfX)
        
        #forward propagation during testing
        Y_hat_test = self.forward_test(tfX)
        
        #test predictions
        predictions = self.predict(Y_hat_test)
        
        #cost function
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = tfY,logits = Y_hat)
        )
        
        #test cost function
        valid_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = tfY,logits = Y_hat_test)
        )
        
        #define RMSProp optimizer
        optimizer = tf.train.RMSPropOptimizer(lr,decay=decay,momentum=mu).minimize(cost)
        
        
        #training loop
        train_loss = []
        valid_loss = []
        valid_errs = []
        n_batches = n//batch_size
        
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                X_train,Y_train = shuffle(X_train,Y_train)
                for n_batch in range(n_batches):
                    X_batch = X_train[(n_batch*batch_size):((n_batch*batch_size)+batch_size)]
                    Y_batch = Y_train[(n_batch*batch_size):((n_batch*batch_size)+batch_size)]
                    feeder = {tfX:X_batch,tfY:Y_batch}
                    
                    #run the optimizer
                    session.run(optimizer,feed_dict=feeder)
                    #at each print_period
                    if n_batch%print_period == 0:
                        l = session.run(cost,feed_dict = feeder)
                        train_loss.append(l)
                        valid_l = session.run(valid_cost,feed_dict={tfX:X_valid,tfY:Y_valid})
                        valid_loss.append(valid_l)
                        valid_preds = session.run(predictions,feed_dict={tfX:X_valid})
                        #print(valid_preds[:20])
                        #print(np.argmax(Y_valid,axis=1)[:20])
                        valid_err = self.err_rate(np.argmax(Y_valid,axis=1),valid_preds)
                        valid_errs.append(valid_err)
                        print('Epoch : %d,Batch : %d,train loss : %.4f,valid loss : %.4f,valid err: %.4f' %(epoch,n_batch,l,valid_l,valid_err))
        if show_fig:
            plt.plot(train_loss,label='Training loss')
            plt.plot(valid_loss,label='Validation loss')
            plt.legend()
                        
    
    #forward propagation functions
    def forward_train(self,X,decay = 0.9):
        Z = X
        Z = tf.nn.dropout(Z,self.keep_probs[0])
        for params,p in zip(self.all_params,self.keep_probs[1:]):
            Z = tf.matmul(Z,params['W'])
            #batch norm
            batch_mean,batch_var = tf.nn.moments(Z,[0])
            update_rn_mean = tf.assign(
                params['rn_mean'],
                params['rn_mean']*decay + (1.0-decay)*batch_mean
            )
            update_rn_var = tf.assign(
                params['rn_var'],
                params['rn_var']*decay + (1.0-decay)*batch_var
            )
            with tf.control_dependencies([update_rn_mean,update_rn_var]):
                Z = tf.nn.batch_normalization(Z,batch_mean,batch_var,params['beta'],params['gamma'],1e-8)
            Z = tf.nn.relu(Z)
            Z = tf.nn.dropout(Z,p)
        return tf.matmul(Z,self.W)+self.b
            
    
    def forward_test(self,X):
        Z = X
        for params in self.all_params:
            Z = tf.matmul(Z,params['W'])
            Z = tf.nn.batch_normalization(Z,params['rn_mean'],params['rn_var'],params['beta'],params['gamma'],1e-8)
            Z = tf.nn.relu(Z)
        return tf.matmul(Z,self.W)+self.b
    
    #prediction function
    def predict(self,Y_hat):
        return tf.argmax(Y_hat,1)
    
    #compute error rate
    def err_rate(self,labels,preds):
        return (labels != preds).mean()
    
    