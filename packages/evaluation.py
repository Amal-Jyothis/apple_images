import numpy as np
import torch
import torch.nn as nn
from ignite.metrics import *
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def fid(gen_data_path, real_data_path):
    '''
    calculates fid score for real and generated data
    '''

    model = torchvision.models.inception_v3(weights='DEFAULT', transform_input=False)
    # model = torch.hub.load('pytorch/vision:v0.13.0', 'inception_v3')
    model.eval()
    model.fc = torch.nn.Identity()

    #Parameters
    image_size = 299
    batch_size = 200

    #image transformation details
    image_transforms = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    gen_dataset = datasets.ImageFolder(root=gen_data_path, transform=image_transforms)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=True)

    real_dataset = datasets.ImageFolder(root=real_data_path, transform=image_transforms)
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    gen_features = np.zeros((len(gen_dataset), 2048))
    real_features = np.zeros((len(real_dataset), 2048))

    with torch.no_grad():
        for i, (images, _) in enumerate(gen_dataloader):
            gen_features = model(images).numpy()

    with torch.no_grad():
        for i, (images, _) in enumerate(real_dataloader):
            real_features = model(images).numpy()
    
    gen_mean = np.mean(gen_features, axis=0)
    real_mean = np.mean(real_features, axis=0)

    gen_covariance = np.cov(gen_features, rowvar=False)
    real_covariance = np.cov(real_features, rowvar=False)

    cov_eig = np.real(np.linalg.eigvals(np.dot(real_covariance, gen_covariance)))
    cov_sqrt = np.sqrt(np.clip(cov_eig, 0, None))

    fid_score = np.linalg.norm(real_mean - gen_mean) + np.trace(real_covariance + gen_covariance - 2 * cov_sqrt)

    return fid_score
    
def inception_score(gen_data):
    '''
    calculates inception score for generated data
    '''
    metric = InceptionScore()
    metric.update(gen_data)
    inception_score = metric.compute()
    return inception_score

def fid_score(gen_data, real_data):
    '''
    calculates fid score for real and generated data
    '''
    metric = FID()
    metric.update(gen_data)
    fid_score = metric.compute()
    return fid_score


def main():
    pass

if __name__ == '__main__':
    main()