import time
import datetime
import numpy

from packages.data_collect import*
from packages.gan import*
from packages.evaluation import*
from packages.img_generation import*

if __name__ == "__main__":
    start_time = time.time()
    print('Run strated at:', datetime.datetime.now())

    '''
    Start of input data extraction
    '''
    input_path = r'input_data\input_images'
    save_input_img_path = r'input_data\input_saved_images\apples'

    dataloader = get_data(input_path)
    save_input_data(dataloader, save_input_img_path)    

    '''
    Train gan model
    '''
    model_save_path = r'output\saved_model\model.pth'
    image_save_path = r'output\generated_images\apples'
    hyperparameters = {'learning_rate_G': 5e-4,
                       'learning_rate_D': 5e-4,
                       'g_iter': 1,
                       'd_iter': 1,
                       'latent_size': 100,
                       'reg_G': 0,
                       'reg_D': 0}

    gan(dataloader, model_save_path, image_save_path, **hyperparameters)

    image_generation(model_save_path, image_save_path, hyperparameters['latent_size'], eg_nos_latent=100)

    input_img_path = r'input_data\input_saved_images'
    generated_image_path = r'output\generated_images'
    
    print('FID Score: ', fid(generated_image_path, input_img_path))
    #print('FID Score: ', fid_score(gen_data, real_data))
    #print('Inception Score: ', inception_score(gen_data))

    end_time = time.time()
    print('Time taken:', end_time - start_time)
    
