import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

def get_data(folder_path):

    #Parameters
    image_size = 64
    batch_size = 200

    #image transformation details
    image_transforms = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  
            ])
    
    dataset = datasets.ImageFolder(root=folder_path, transform=image_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def save_input_data(dataloader, save_folder_path):

    images,_  = next(iter(dataloader))
    j = 0
    
    for img in images:
        img = img.detach().numpy()
        
        save_path = os.path.join(save_folder_path, f"input_image_{j}.png")
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img.transpose(1, 2, 0) * 0.5 + 0.5) 
        plt.savefig(save_path)
        plt.close()
        j = j + 1

    

                

