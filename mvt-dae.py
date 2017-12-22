#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:13:01 2017

@author: abdala
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:58:15 2017

@author: abdala
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mmd_loss_2 as mmd
import config
tf.reset_default_graph()
def datapreparation():
      t_cnn1 = pd.read_csv(config.t_file_name_cnn1)
      t_cnn2 = pd.read_csv(config.t_file_name_cnn2)
      t_label = pd.read_csv(config.t_file_label)
      t_label = t_label.iloc[:,1:].values
      
      s_cnn1 = pd.read_csv(config.s_file_name_cnn1)
      s_cnn2 = pd.read_csv(config.s_file_name_cnn2)
      s_label = pd.read_csv(config.s_file_label)
      s_label = s_label.iloc[:,1:].values
      
      s_cnn1_input = s_cnn1.iloc[:,1:].values
      s_cnn2_input = s_cnn2.iloc[:,1:].values
      
      t_cnn1_input = t_cnn1.iloc[:,1:].values
      t_cnn2_input = t_cnn2.iloc[:,1:].values
      
      s_cnn1_input = MinMaxScaler().fit_transform(s_cnn1_input)
      s_cnn2_input = MinMaxScaler().fit_transform(s_cnn2_input)
      
      t_cnn1_input = MinMaxScaler().fit_transform(t_cnn1_input)
      t_cnn2_input = MinMaxScaler().fit_transform(t_cnn2_input)
      return s_cnn1_input,s_cnn2_input,t_cnn1_input,t_cnn2_input,s_label, t_label

s_cnn1_input,s_cnn2_input,t_cnn1_input,t_cnn2_input,s_label, t_label = datapreparation()
alpha = 0.001

#parameter for the source domain with 3 views
S_num_input_1 = 2048
S_num_input_2 = 2048

S_hidden_1 =100
S_hidden_2 = 3
S_hidden_3=100
S_num_output_1 = 2048
S_num_output_2 = 2048

# parameter for the target domain with 3 views
T_num_input_1 = 2048
T_num_input_2 = 2048

T_hidden_1 =100
T_hidden_2 = 3
T_hidden_3=100
T_num_output_1 = 2048
T_num_output_2 = 2048
learning_rate = 0.0001
# Let's define our place folder for SOurce domain
X_Source_1 =tf.placeholder(tf.float32,shape = [None ,S_num_input_1])
X_Source_2 =tf.placeholder(tf.float32,shape = [None ,S_num_input_2])

Y_Source =tf.placeholder(tf.float32,shape = [None ,S_hidden_2])
z_S = tf.placeholder(tf.int64, shape=[None])

# Let's define our place folder for Target domain
X_Target_1 =tf.placeholder(tf.float32,shape = [None ,T_num_input_1])
X_Target_2 =tf.placeholder(tf.float32,shape = [None ,T_num_input_2])
z_T = tf.placeholder(tf.int64, shape=[None])

def fc(list_X, list_shape):
      list_tensor = []
      i = 0
      for X,shape in zip(list_X, list_shape):
            initializer = tf.truncated_normal_initializer(stddev=0.01, mean=0)
            var_name = 'kernel_'+str(i)
            weight = tf.get_variable(var_name, shape, initializer=initializer)
            list_tensor.append(tf.matmul(X, weight))
            i = i+1
      bias = tf.get_variable('bias', shape[-1:], initializer=initializer)
      final_tensor = tf.reduce_sum(tf.convert_to_tensor(list_tensor),axis = 0)
      return final_tensor + bias

def fc_sigmoid(X, shape):
      return tf.nn.sigmoid(fc(X, shape))

def fc_relu(X, shape):
      return tf.nn.relu(fc(X, shape))

def build_train():
      #embedding layer
      with tf.variable_scope('embedding') as scope:
            S_hid_layer_1 = fc_relu([X_Source_1,X_Source_2],
                              [(S_num_input_1,S_hidden_1),(S_num_input_2,S_hidden_1)])
            scope.reuse_variables()
            T_hid_layer_1 = fc_relu([X_Target_1,X_Target_2],
                              [(T_num_input_1,T_hidden_1),(T_num_input_2,T_hidden_1)])
      
      with tf.variable_scope('label') as scope:
            S_hid_layer_2 = fc_relu([S_hid_layer_1],[(S_hidden_1,S_hidden_2)])
            scope.reuse_variables()
            logit_S = fc([S_hid_layer_1],[(S_hidden_1,S_hidden_2)])
            T_hid_layer_2 = fc_relu([T_hid_layer_1],[(T_hidden_1,T_hidden_2)])
            logit_T = fc([T_hid_layer_1],[(T_hidden_1,T_hidden_2)])
      
      with tf.variable_scope ('rec_embdedding') as scope:
            S_hid_layer_3 = fc_relu([S_hid_layer_2], [(S_hidden_2,S_hidden_3)])
            scope.reuse_variables()
            T_hid_layer_3 = fc_relu([T_hid_layer_2], [(T_hidden_2,T_hidden_3)])
            
      with tf.variable_scope('rec_input_1') as scope:
            S_out_1 = fc_relu([S_hid_layer_3],[(S_hidden_3,S_num_output_1)])
            scope.reuse_variables()
            T_out_1 = fc_relu([T_hid_layer_3],[(T_hidden_3,T_num_output_1)])
            
      with tf.variable_scope('rec_input_2') as scope:
            S_out_2 = fc_relu([S_hid_layer_3],[(S_hidden_3,S_num_output_2)])
            scope.reuse_variables()
            T_out_2 = fc_relu([T_hid_layer_3],[(T_hidden_3,T_num_output_2)])
      
      pred_S = tf.nn.softmax(logit_S)
      pred_cls_S = tf.argmax(pred_S,1)
      correct_prediction_S = tf.equal(pred_cls_S, z_S)
      accuracy_S = tf.reduce_mean(tf.cast(correct_prediction_S, tf.float32))
      
      pred_T = tf.nn.softmax(logit_T)
      pred_cls_T = tf.argmax(pred_T,1)
      correct_prediction = tf.equal(pred_cls_T, z_T)
      accuracy_T = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
      epsilon = tf.constant(value=0.1,shape = [S_hidden_2])
      logit_S = logit_S + epsilon
      Softmax_layer_S = tf.nn.softmax(logit_S)
      cross_entropy = -tf.reduce_sum(Y_Source * tf.log(Softmax_layer_S), reduction_indices=[1])
      
      # KL divergence between distributions of source and target domains
      dist_h = tf.reduce_mean(S_hid_layer_1, axis=0)
      dist_t = tf.reduce_mean(T_hid_layer_1, axis=0)

      KL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_h, labels=dist_h/dist_t))
      KL += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dist_t, labels=dist_t/dist_h))
    

      loss_logit =tf.reduce_mean(cross_entropy)
      loss_S_rec =tf.reduce_mean(tf.square(X_Source_1 - S_out_1)) + tf.reduce_mean(tf.square(X_Source_2 - S_out_2))
      loss_T_rec =tf.reduce_mean(tf.square(X_Target_1 - T_out_1)) + tf.reduce_mean(tf.square(X_Target_2 - T_out_2)) 
      mmd_loss,A ,B,C,K_XX, K_XY, K_YY = mmd.join_mix_rbf_mmd2([S_hid_layer_1,S_hid_layer_2,S_hid_layer_3],
                                  [T_hid_layer_1,T_hid_layer_2,T_hid_layer_3],num_layer=3,Ns=1500,Nt=800)
      
      L2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
      
      mvt_dae_loss = loss_logit +mmd_loss+ 0.001*loss_T_rec + 0.001*loss_S_rec +0.01*L2_loss 
      
      rec_loss = loss_T_rec + loss_S_rec 
                         
      optimizer = tf.train.AdamOptimizer(learning_rate)
      train_mvt_dae_loss = optimizer.minimize(mvt_dae_loss)
      train_auto_encoder = optimizer.minimize(rec_loss)
      init = tf.global_variables_initializer()
       
      with tf.Session() as sess:
          sess.run(init)
          # Train the auto-encoder first
          print('We are are reconstucting the input , please kindly wait....')
          for iters in range(100):
              _,rec_loss_ =sess.run([train_auto_encoder,rec_loss], feed_dict = 
                                                          {
                                                           X_Source_1:s_cnn1_input,
                                                           X_Source_2:s_cnn2_input,
                                                           Y_Source:s_label,
                                                           X_Target_1:t_cnn1_input,
                                                           X_Target_2:t_cnn2_input
                                                          })
              if iters%10 == 0:
                     print("Iter %d"%iters+" rec_loss = %f"%rec_loss_)
                     #print(temp_)
          print("Reconstruction finished finished....")
          
          for iters in range(200):
              A_,B_,C_,K_XX_, K_XY_, K_YY_,mmd_,_,loss =sess.run([A ,B,C,K_XX, K_XY, K_YY,mmd_loss,train_mvt_dae_loss,mvt_dae_loss], feed_dict = 
                      {
                       X_Source_1:s_cnn1_input,
                       X_Source_2:s_cnn2_input,
                       Y_Source:s_label,
                       X_Target_1:t_cnn1_input,
                       X_Target_2:t_cnn2_input
                      })
              if iters%10 == 0:
                     print("Iter %d"%iters+" loss = %f"%loss)
                     #print(temp_)
          print("Optimization finished....")
          
          acc_train_T = sess.run([accuracy_T], feed_dict=
                         {
                                X_Target_1:t_cnn1_input,
                                X_Target_2:t_cnn2_input,
                                z_T : np.argmax(t_label[:,:],axis=1),
                                X_Source_1:s_cnn1_input,
                                X_Source_2:s_cnn2_input,
                                z_S : np.argmax(s_label[:,:],axis=1)
                         })
      
          pred_T = sess.run([pred_T], feed_dict=
                         {
                                X_Target_1:t_cnn1_input,
                                X_Target_2:t_cnn2_input,
                                z_T : np.argmax(t_label[:,:],axis=1),
                                X_Source_1:s_cnn1_input,
                                X_Source_2:s_cnn2_input,
                                z_S : np.argmax(s_label[:,:],axis=1)
                         })
      
          pred_cls_T = sess.run([pred_cls_T], feed_dict=
                         {
                                X_Target_1:t_cnn1_input,
                                X_Target_2:t_cnn2_input,
                                z_T : np.argmax(t_label[:,:],axis=1),
                                X_Source_1:s_cnn1_input,
                                X_Source_2:s_cnn2_input,
                                z_S : np.argmax(s_label[:,:],axis=1)
                         })
            
          
       ########################################################
         
          acc_train_S = sess.run([accuracy_S], feed_dict=
                         {
                                X_Target_1:t_cnn1_input,
                                X_Target_2:t_cnn2_input,
                                z_T : np.argmax(t_label[:,:],axis=1),
                                X_Source_1:s_cnn1_input,
                                X_Source_2:s_cnn2_input,
                                z_S : np.argmax(s_label[:,:],axis=1)
                         })
      
          pred_cls_S = sess.run([pred_cls_S], feed_dict=
                         {
                                X_Target_1:t_cnn1_input,
                                X_Target_2:t_cnn2_input,
                                z_T : np.argmax(t_label[:,:],axis=1),
                                X_Source_1:s_cnn1_input,
                                X_Source_2:s_cnn2_input,
                                z_S : np.argmax(s_label[:,:],axis=1)
                         })
          pred_S = sess.run([pred_S], feed_dict=
                         {
                                X_Target_1:t_cnn1_input,
                                X_Target_2:t_cnn2_input,
                                z_T : np.argmax(t_label[:,:],axis=1),
                                X_Source_1:s_cnn1_input,
                                X_Source_2:s_cnn2_input,
                                z_S : np.argmax(s_label[:,:],axis=1)
                         })
      return acc_train_T , acc_train_S, pred_T,pred_S,pred_cls_T,pred_cls_S

acc_train_T , acc_train_S, pred_T,pred_S,pred_cls_T,pred_cls_S = build_train()
 