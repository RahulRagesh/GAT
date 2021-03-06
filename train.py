from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys 

import tensorflow as tf

from utils import *
from models import GAT


import optuna 
from optuna.samplers import TPESampler

import random

class TuneGAT():
    def __init__(self, dataset):
        ###################################################################################################
        # Prepare data
        ###################################################################################################
        self.adj, self.features, self.labels, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = load_data(dataset)

        self.unlabeled_mask = ~(self.train_mask|self.val_mask|self.test_mask)
        self.y_unlabeled = self.labels.copy()
        self.y_unlabeled[~self.unlabeled_mask] = 0

        self.adj_t = preprocess_adj(self.adj)
        self.features = preprocess_features(self.features)
        ###################################################################################################
        
        
        ###################################################################################################
        # Params
        ###################################################################################################
        self.epochs = 500
        self.early_stopping = 25
        self.seed = 123
        self.verbose = True
        ###################################################################################################

    def __call__(self, trial):
        print('TRIAL N0    - %d'%trial.number)
        ###################################################################################################
        # Hyperparameters
        ###################################################################################################
        self.configs = {}
        self.configs['seed'] = self.seed
        
        ################################################################
        # Initialize Hyperparameters for search
        # Comment This block when specifying hyperparameters
        self.configs['learning_rate'] = trial.suggest_uniform('learning_rate',0,1)
        self.configs['weight_decay'] = trial.suggest_uniform('weight_decay',0,1)
        self.configs['model_dropout'] = trial.suggest_uniform('model_dropout',0,1)
        self.configs['attention_dropout'] = trial.suggest_uniform('attention_dropout',0,1)
        ################################################################

        '''
        ################################################################
        # Specifying hyperparamters
        # Comment this block when searching for hyperparameters
        # Set n_trials to 1
        self.configs['learning_rate'] = 0.005
        self.configs['weight_decay'] = 5e-4
        self.configs['model_dropout'] = 0.6
        self.configs['attention_dropout'] = 0.6
        ################################################################
        '''
        
        self.configs['hidden_dims'] = 8
        self.configs['num_heads'] = 8
        self.configs['num_heads_output'] = 1
        ###################################################################################################



        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            ###################################################################################################
            # Set random seed
            ###################################################################################################        
            tf.set_random_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            ###################################################################################################

            # Define placeholders
            self.placeholders = {
                'adj': tf.sparse_placeholder(tf.float32, name='adj'),
                'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64), name='features'),
                'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1]), name='labels'),
                'labels_mask': tf.placeholder(tf.int32,name='labels_mask'),
                'model_dropout': tf.placeholder_with_default(0., shape=(), name='model_dropout'),
                'attention_dropout': tf.placeholder_with_default(0., shape=(), name='attention_dropout')
            }

            # Create model
            self.model = GAT(self.configs, self.placeholders, input_dim=self.features[2][1], logging=True)

            # Init variables
            sess.run(tf.global_variables_initializer())

            patience_count = 0
            best_val_accuracy = 0.0

            # Train model
            for epoch in range(self.epochs):
                # Construct feed dictionary
                feed_dict = construct_feed_dict(self.features, self.adj_t, self.y_train, self.train_mask, self.placeholders)
                feed_dict.update({self.placeholders['model_dropout']: self.configs['model_dropout']})
                feed_dict.update({self.placeholders['attention_dropout']: self.configs['attention_dropout']})

                # Training step
                outs = sess.run([self.model.opt_op, self.model.loss, self.model.accuracy], feed_dict=feed_dict)

                # Evaluate
                train_loss, train_acc, _ = evaluate(self.model, sess, self.features, self.adj_t, self.y_train, self.train_mask, self.placeholders)
                val_loss, val_acc, _ = evaluate(self.model, sess, self.features, self.adj_t, self.y_val, self.val_mask, self.placeholders)
                test_loss, test_acc, _ = evaluate(self.model, sess, self.features, self.adj_t, self.y_test, self.test_mask, self.placeholders)
                unlabeled_loss, unlabeled_acc, _ = evaluate(self.model, sess, self.features, self.adj_t, self.y_unlabeled, self.unlabeled_mask, self.placeholders)

                if self.verbose:
                    # Print results
                    print('[Epoch %03d] Loss - %0.04f\t Train Accuracy - %0.04f\t Val Accuracy - %0.04f'%(epoch+1, outs[1], train_acc, val_acc))

                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    patience_count = 0            
                    trial.set_user_attr('Train Accuracy', train_acc)
                    trial.set_user_attr('Train Loss', train_loss)
                    trial.set_user_attr('Val Accuracy', val_acc)
                    trial.set_user_attr('Val Loss', val_loss)
                    trial.set_user_attr('Test Accuracy', test_acc)
                    trial.set_user_attr('Test Loss', test_loss)
                    trial.set_user_attr('Unlabeled Accuracy', unlabeled_acc)
                    trial.set_user_attr('Unlabeled Loss', unlabeled_loss)   
                    
                    '''
                    #############################################################################################################
                    #  Fetch Attention Weights to analyze
                    #############################################################################################################
                    feed_dict.update({self.placeholders['model_dropout']: 0.})
                    feed_dict.update({self.placeholders['attention_dropout']: 0.})
                    
                    layer_1_attention_matrices = [sess.run(self.model.layers[0].attention_matrices[i], feed_dict=feed_dict) for i in range(self.configs['num_heads'])]
                    layer_1_attention_matrices = [sparse_tensor_to_coo(sp_mat) for sp_mat in layer_1_attention_matrices]
                    trial.set_user_attr('layer_1_attention_matrices', layer_1_attention_matrices)   
                    
                    layer_2_attention_matrices = [sess.run(self.model.layers[1].attention_matrices[i], feed_dict=feed_dict) for i in range(self.configs['num_heads_output'])]
                    layer_2_attention_matrices = [sparse_tensor_to_coo(sp_mat) for sp_mat in layer_2_attention_matrices]                    
                    trial.set_user_attr('layer_2_attention_matrices', layer_2_attention_matrices)   
                    #############################################################################################################                    
                    '''
                else:
                    patience_count = patience_count + 1 

                if patience_count >= self.early_stopping: #Early stopping
                    break 
            
            print()            
            print('Train Accuracy       - %0.04f'%(trial.user_attrs['Train Accuracy']))
            print('Val Accuracy         - %0.04f'%(trial.user_attrs['Val Accuracy']))
            print('Test Accuracy        - %0.04f'%(trial.user_attrs['Test Accuracy']))
            print('Unlabeled Accuracy  - %0.04f'%(trial.user_attrs['Unlabeled Accuracy']))
            print()
        
        return trial.user_attrs['Val Accuracy']
           

dataset = sys.argv[1]

optuna.logging.disable_default_handler()
sampler = TPESampler(seed=123)
model_study = optuna.create_study(direction='maximize',sampler=sampler)
model_objective = TuneGAT(dataset)
model_study.optimize(model_objective, n_trials=500, callbacks = [model_trial_log_callback])

trial = get_best_trial(model_study) 

print("Best Hyperparameters")
for key, value in trial.params.items():
    print('%s : %s'%(key,str(value)))


print('BEST TRIAL N0    - %d'%trial.number)
print('Train Accuracy       - %0.04f'%(trial.user_attrs['Train Accuracy']))
print('Train Loss           - %0.04f'%(trial.user_attrs['Train Loss']))
print('Val Accuracy         - %0.04f'%(trial.user_attrs['Val Accuracy']))
print('Val Loss             - %0.04f'%(trial.user_attrs['Val Loss']))
print('Test Accuracy        - %0.04f'%(trial.user_attrs['Test Accuracy']))
print('Test Loss            - %0.04f'%(trial.user_attrs['Test Loss']))
print('Unlabeled Accuracy  - %0.04f'%(trial.user_attrs['Unlabeled Accuracy']))
print('Unlabeled Loss      - %0.04f'%(trial.user_attrs['Unlabeled Loss']))

model_df = model_study.trials_dataframe()
model_df.to_csv('config_metrics_%s.csv'%dataset,index=False)

# To access attention weights, uncomment the attention section above  
# and use trial.user_attrs['layer_1_attention_matrices']