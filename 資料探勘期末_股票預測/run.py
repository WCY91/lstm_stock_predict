import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from keras.utils import plot_model

def plot_results(predicted_data, true_data,p1,p2):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    #plt.show()
    plt.savefig(f'./result/results4_{p1}_{p2}.png')
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len,p1,p2):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.legend()
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    #plt.show()
    plt.savefig(f'./result/results_multiple_{p1}_{p2}.png')
    plt.show()

def main():
    
    configs = json.load(open('config_4.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    
    model = Model()
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print (x.shape)
    print (y.shape)
    model.build_model(configs)
	
    model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	
   
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
   
    predictions_multiseq = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    predictions_pointbypoint = model.predict_point_by_point(x_test,debug=True)        
    
    #plot_results_multiple(predictions_multiseq, y_test, configs['data']['sequence_length'],configs['data']['columns'][0],configs['data']['columns'][1])
    plot_results(predictions_pointbypoint, y_test,configs['data']['columns'][0],configs['data']['columns'][1])
    
if __name__ == '__main__':
    main()