import torch.nn as nn
import torch
import numpy as np
from src.utils.utils import batch_dft_to_audio, plot_loss_function



def train_autoencoder(autoencoder, train_loader, optimizer,
              model_name, models_path, num_epochs=10,
              device='cpu', lbmda = 0.5, losses_list = None,
              epoch_start = 0):
    autoencoder = autoencoder.to(device)
    autoencoder.train() # Set the generator in train mode

    total_step = len(train_loader)
    
    if type(losses_list) == type(None):
        losses_list = []

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # Iterate over epochs
    for epoch in range(epoch_start, num_epochs):
        print(f"Starting epoch {epoch}...")
        # Iterate the dataset
        loss_sum = 0
        n_batches = 0

        for i, dfts in enumerate(train_loader):
            # Get batch of samples and labels
            dfts_device = dfts.to(device)
            
            # Forward pass
            # Generate random images with the generator
            synth_dfts = autoencoder.forward(dfts_device)
            
            # Generator loss
            cur_l1_loss = lbmda * l1_loss(synth_dfts, dfts_device)
            cur_mse_loss = mse_loss(synth_dfts, dfts_device)

            autoencoder_loss = cur_mse_loss + cur_l1_loss

            optimizer.zero_grad()
            autoencoder_loss.backward() # Necessary to not erase intermediate variables needed for computing disc_loss gradient
            optimizer.step()
                
            loss_sum += autoencoder_loss.cpu().item()

            n_batches += 1

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]. "
                      f"Step [{i+1}/{total_step}]. "
                      f"Loss: {round(loss_sum/n_batches, 5)}.")
        
        loss_avg = loss_sum/n_batches
        print(f"Epoch [{epoch+1}/{num_epochs}]. "
              f"Step [{i+1}/{total_step}]. "
              f"Loss: {round(loss_avg, 5)}.")
        
        # Listen batch
        _ = batch_dft_to_audio(synth_dfts, listen=True)
        
        # Plot Loss function
        plot_loss_function(losses_list, "Current Training Loss")
        
        # Save model        
        model_checkpoint_filename = models_path / f"{model_name}__epoch_{epoch}.ckpt"
        torch.save(autoencoder.state_dict(),
                   model_checkpoint_filename)
        # Save model loss
        model_loss_filename = models_path / f"{model_name}__epoch_{epoch}_loss.npy"
        losses_list.append(loss_avg)
        np.save(model_loss_filename, np.array(losses_list))
        
    # Save final model
    model_filename = models_path / f"{model_name}__final.ckpt"
    torch.save(autoencoder.state_dict(),
                model_filename)
    # Save loss history
    model_loss_filename = models_path / f"{model_name}__final_loss.npy"
    np.save(model_loss_filename, np.array(losses_list))
    
          
    return model_filename, model_loss_filename, losses_list
