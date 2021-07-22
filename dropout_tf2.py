#ANN with dropout using tensorflow
from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.utils import shuffle
from util import init_weights

class ANN:
    def __init__(self,hidden_layer_sizes,keep_probs):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.keep_probs = keep_probs
        
        #initiate parameters except the first and final layer
        self.all_params = []
        m1 = self.hidden_layer_sizes[0]
        for m2 in hidden_layer_sizes[1:]:
            w_init,b_init = init_weights(m1,m2)
            W = tf.Variable(w_init)
            b = tf.Variable(b_init)
            self.all_params += [(W,b)]
            m1=m2
        
    def fit(self,X,Y,batch_size,epochs,valid_size = 1000,lr = 0.001,reg = 0.1,decay = 0.9 , mu = 0.99,eps=1e-8, print_period=50,show_fig=True):
        
        lr = np.float32(lr)
        reg = np.float32(reg)
        mu = np.float32(mu)
        decay = np.float32(decay)
        eps = np.float32(eps)
        
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        X,Y = shuffle(X,Y)
        X_train,X_valid = X[:-valid_size,:],X[-valid_size:,:]
        Y_train,Y_valid = Y[:-valid_size,:],Y[-valid_size:,:]
        
        n,m = X_train.shape
        p = Y_train.shape[1]
        #add first and last layer weights
        w0_init,b0_init = init_weights(m,self.hidden_layer_sizes[0])
        W0 = tf.Variable(w0_init)
        b0 = tf.Variable(b0_init)
        self.all_params = [(W0,b0)]+self.all_params
        
        wf_init,bf_init = init_weights(self.hidden_layer_sizes[-1],p)
        self.W = tf.Variable(wf_init)
        self.b = tf.Variable(bf_init)
        self.all_params = self.all_params + [(self.W,self.b)]
        
        #create placeholders
        tfX = tf.placeholder(tf.float32,shape=(None,m),name='X')
        tfY = tf.placeholder(tf.float32,shape=(None,p),name='Y')
        
        #forward propagation for training
        Y_hat = self.forward_train(tfX)
        
        #forward propagation during testing
        Y_hat_test = self.forward_test(tfX)
        predictions = self.predict(Y_hat_test)
        
        #training cost computation
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tfY,logits = Y_hat)
        )
        
        #testing cost computation
        test_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tfY,logits = Y_hat_test)
        )
        
        #optimizer : training function
        optimizer = tf.train.RMSPropOptimizer(lr,momentum=mu,decay=decay).minimize(cost)
        
        ##create the training loop
        n_batches = n//batch_size
        train_loss = []
        valid_loss = []
        valid_errs = []
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                for n_batch in range(n_batches):
                    X_batch = X_train[(n_batch*batch_size):((n_batch*batch_size)+batch_size),:]
                    Y_batch = Y_train[(n_batch*batch_size):((n_batch*batch_size)+batch_size),:]
                    
                    feeder = {tfX : X_batch,tfY: Y_batch}
                    session.run(optimizer,feed_dict=feeder)
                    
                    if n_batch%print_period == 0:
                        l = session.run(cost,feed_dict = feeder)
                        train_loss.append(l)
                        valid_l = session.run(test_cost,feed_dict={tfX:X_valid,tfY:Y_valid})
                        valid_loss.append(valid_l)
                        
                        valid_preds = session.run(predictions,feed_dict={tfX : X_valid})
                        valid_err = (valid_preds != np.argmax(Y_valid,axis=1)).mean()
                        valid_errs.append(valid_err)
                        print('Epoch : %d,Batch : %d,Training loss : %.4f,Valid loss : %.4f,valid err : %.4f' %(epoch,n_batch,l,valid_l,valid_err))
                        
        if show_fig:
            plt.plot(train_loss,label='Training Loss')
            plt.legend()
                        
        
        
        
        
    def forward_train(self,X):
        Z = X
        Z = tf.nn.dropout(Z,self.keep_probs[0])
        for weights,p in zip(self.all_params[:-1],self.keep_probs[1:]):
            w,b = weights[0],weights[1]
            Z = tf.nn.relu(tf.matmul(Z,w)+b)
            Z = tf.nn.dropout(Z,p)
        Z = tf.matmul(Z,self.W) + self.b
        return Z
    
    def forward_test(self,X):
        Z = X
        for weights in self.all_params:
            w,b = weights[0],weights[1]
            Z = tf.nn.relu(tf.matmul(Z,w)+b)
        return Z
        
        
    def predict(self,Y_hat):
        return tf.argmax(Y_hat,1)