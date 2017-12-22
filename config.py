# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:37:20 2017

@author: XQing
"""

t_file_name_cnn1 = 'C:/Users/XQing/Desktop/tf/data_images/source_2/inception_features.csv'
t_file_name_cnn2 = 'C:/Users/XQing/Desktop/tf/data_images/source_2/restNet_features.csv'
t_file_label = 'C:/Users/XQing/Desktop/tf/data_images/source_2/label_source_2.csv'

# Importing Target domain data Beijing
s_file_name_cnn1 = 'C:/Users/XQing/Desktop/tf/data_images/source_1/inception_features.csv'
s_file_name_cnn2 = 'C:/Users/XQing/Desktop/tf/data_images/source_1/restNet_features.csv'
s_file_label = 'C:/Users/XQing/Desktop/tf/data_images/source_1/label_source_1.csv'

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
