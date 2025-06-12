import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class Generator(torch.nn.Module):
    """
    Generator definition
    """
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_size, 1024, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(1024, 1024, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)
    
class Discriminator(torch.nn.Module):
    """
    Discriminator definition
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(32, 32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(32, 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(16, 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(8, 1, 4, 1, 0, bias=False),
            #torch.nn.Sigmoid()    
        )
    
    def forward(self, input):
        return self.main(input)

class model_definition():
    """
    This class defines the optimiser type and the learning rate used for optimization
    """
    def __init__(self, device, latent_size, learning_rate_G, learning_rate_D, reg_G, reg_D, beta_1=0.5, beta_2=0.999):
        betas = (beta_1, beta_2)

        self.model_gen = Generator(latent_size).to(device)
        self.optimizerG = torch.optim.Adam(self.model_gen.parameters(), lr=learning_rate_G, betas=betas)#, weight_decay=reg_G)

        self.model_discr = Discriminator().to(device)
        self.optimizerD = torch.optim.Adam(self.model_discr.parameters(), lr=learning_rate_D, betas=betas)#, weight_decay=reg_D)

def gan(dataloader, model_save_path, image_save_path, **kwargs):
    """
    This function trains a GAN model with the given training data and hyperparameters.

    Parameters:
    x_train: Training data
    **kwargs: Additional hyperparameters
    """

    # define required hyperparameters
    lr_G = float(kwargs.get("learning_rate_G"))
    lr_D = float(kwargs.get("learning_rate_D"))
    g_iter = int(kwargs.get("g_iter"))
    d_iter = int(kwargs.get("d_iter"))
    latent_size = int(kwargs.get("latent_size"))
    reg_G = float(kwargs.get("reg_G"))
    reg_D = float(kwargs.get("reg_D"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_definition(device, latent_size, lr_G, lr_D, reg_G, reg_D)

    print('Training GAN model...')
    training_gan(model, device, dataloader, latent_size, num_epochs=1000, discr_train_iter = d_iter, gen_train_iter = g_iter)

    # image_generation(model, image_save_path, latent_size, 10)

    torch.save(model, model_save_path)

    return 


"""
training model
"""
def training_gan(model, device, dataloader, latent_size, num_epochs = 5, discr_train_iter = 5, gen_train_iter = 1):

    """"
    Initialize loss values for plotting
    """
    loss_plotD = np.zeros(num_epochs)
    loss_plotGD = np.zeros(num_epochs)
    loss_plotG = np.zeros(num_epochs)

    discr_output_mean = np.zeros(num_epochs)
    output_mean = np.zeros(num_epochs)
    output_1_mean = np.zeros(num_epochs)
    output_2_mean = np.zeros(num_epochs)

    epochs = np.arange(0, num_epochs)

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            discr_output_mean_b = 0
            output_mean_b = 0
            output_1_mean_b = 0
            output_2_mean_b = 0
                
            """"" 
            Training discriminator
            """

            for i in range(0, 1):
                model.optimizerD.zero_grad()

                """"
                Calculating D(X) and loss function
                """
                outputs_1 = model.model_discr(images).view(-1, 1).to(device)
                y_train = torch.full((images.size()[0], 1), 1.0)

                """"
                Binary cross entropy loss for discriminator
                """
                # loss_frm_D = torch.nn.BCELoss()(outputs_1, y_train)
                loss_frm_D = -torch.sum(outputs_1)/len(outputs_1)

                """"
                Calculating D(G(z)) and loss function
                """

                z = torch.randn(images.size()[0], latent_size, 1, 1).to(device)
                gen_output = model.model_gen(z).detach()
                outputs_2 = model.model_discr(gen_output).view(-1, 1)
                z_output = torch.full((images.size()[0], 1), 0.0)
                # loss_frm_GD = torch.nn.BCELoss()(outputs_2, z_output)
                loss_frm_GD = torch.sum(outputs_2)/len(outputs_2)

                #Calculating Gradient Penalty term for loss function
                # eps = 0.3
                eps = torch.rand(images.shape[0], 1, 1, 1).to(device)
                eps = eps.expand_as(images)
                
                
                Z_bar = eps*images + (1 - eps)*gen_output.detach().numpy()
                Z_bar = Z_bar.requires_grad_(True)
                Z_bar_pred = model.model_discr(Z_bar).view(-1, 1)
                z_bar_grad = torch.autograd.grad(outputs=Z_bar_pred,
                                                 inputs=Z_bar,
                                                 grad_outputs=torch.ones_like(Z_bar_pred),
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 only_inputs=True)[0]

                total_loss = loss_frm_D + loss_frm_GD + 10*((z_bar_grad.norm(2, dim=1) - 1) ** 2).mean()
                total_loss.backward()
                model.optimizerD.step()

            loss_plotD[epoch] = total_loss

            """
            Training generator
            """

            for j in range(0, 1):
                """
                Calculating D(G(z)) and training
                """

                model.optimizerG.zero_grad()
                z = torch.randn(images.size()[0], latent_size, 1, 1).to(device)
                outputs = model.model_discr(model.model_gen(z)).view(-1, 1)
                output_label = torch.full((images.size()[0], 1), 1.0)
                # loss_frm_G = torch.nn.BCELoss()(outputs, output_label)
                loss_frm_G = -torch.sum(outputs)/len(outputs)
                loss_frm_G.backward()
                model.optimizerG.step()

            loss_plotG[epoch] += loss_frm_G

            if discr_output_mean_b == 0:
                output_mean_b = torch.mean(outputs)
                output_1_mean_b = torch.mean(outputs_1)
                output_2_mean_b = torch.mean(outputs_2)
                discr_output_mean_b = (torch.mean(outputs_1) + torch.mean(outputs_2) + torch.mean(outputs))/3
            else:
                output_mean_b = (output_mean_b + torch.mean(outputs))*0.5
                output_1_mean_b = (output_1_mean_b + torch.mean(outputs_1))*0.5
                output_2_mean_b = (output_2_mean_b + torch.mean(outputs_2))*0.5
                discr_output_mean_b = (discr_output_mean_b + torch.mean(outputs_1) + torch.mean(outputs_2) + torch.mean(outputs))/4


        discr_output_mean[epoch] = discr_output_mean_b
        output_mean[epoch] = output_mean_b
        output_1_mean[epoch] = output_1_mean_b
        output_2_mean[epoch] = output_2_mean_b

    """
    Plotting
    """
    print('Loss of Generator: ', loss_plotG[num_epochs - 1])
    print('Loss of Discriminator: ', loss_plotD[num_epochs - 1])

    plt.plot(epochs, loss_plotD, label='Discriminator Loss')
    # plt.plot(epochs, loss_plotGD, label='GD Loss')
    plt.plot(epochs, loss_plotG, label='Generator Loss')
    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel('Iterations', fontsize='10')
    plt.ylabel('Loss', fontsize='10')
    plt.legend(fontsize='10')
    plt.grid()
    plt.savefig(r'GAN_loss_plot.png', dpi=1000)
    # plt.show()

    # plt.plot(epochs, discr_output_mean, label='Discriminator Output Mean')
    # plt.plot(epochs, output_1_mean, label='D(real)) Output')
    # plt.plot(epochs, output_2_mean, label='D(fake) Output 1')
    # plt.plot(epochs, output_mean, label='D(fake) Output 2')
    # plt.legend()
    # plt.show()


def image_generation(model, image_save_path, latent_size, eg_nos_latent = 10):

    """
    Generating the new fake data
    """

    z_test = torch.randn(eg_nos_latent, latent_size, 1, 1)

    outputs = model.model_gen(z_test)
    outputs = outputs.detach().numpy()

    for i in range(outputs.shape[0]):
        save_path = os.path.join(image_save_path, f"generated_image_{i}.png")
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(outputs[i].transpose(1, 2, 0) * 0.5 + 0.5)  
        plt.savefig(save_path)
        plt.close()

    return outputs