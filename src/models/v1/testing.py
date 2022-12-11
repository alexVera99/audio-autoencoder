import torch
import torch.nn as nn

from src.utils.utils import batch_dft_to_audio


def testing(autoencoder: torch.nn.Module, 
            testing_data_loader: torch.utils.data.Dataset, 
            device: torch.device,
            lbmda: float):
    """Test the model with testing data

    Args:
        autoencoder (torch.nn.Module): model to test
        testing_data_loader (torch.utils.data.Dataset): testing dataloader
        device (torch.device): device to run the testing
        lbmda (float): weight to L1 loss
    """
    autoencoder.eval()
    
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    
    total_step = len(testing_data_loader)
    
    loss_sum = 0
    n_batches = 0
    
    print("Running testing...")
    with torch.no_grad(): # inference
        for i, dfts in enumerate(testing_data_loader):
                # Get batch of samples
                dfts_device = dfts.to(device)
                
                # Forward pass
                # Generate random images with the generator
                synth_dfts = autoencoder.forward(dfts_device)
                
                # Generator loss
                cur_l1_loss = lbmda * l1_loss(synth_dfts, dfts_device)
                cur_mse_loss = mse_loss(synth_dfts, dfts_device)

                autoencoder_loss = cur_mse_loss + cur_l1_loss
                    
                loss_sum += autoencoder_loss.cpu().item()

                n_batches += 1

                if (i+1) % 100 == 0:
                    print(f"Step [{i+1}/{total_step}]. "
                        f"Loss: {round(loss_sum/n_batches, 5)}.")
    
    # Listen batch
    print("Original DFTs and Audios")
    _ = batch_dft_to_audio(dfts_device, listen=True, show_spectrogram=True, listen_max=2)
    print("Synthesized DFTs and Audios")
    _ = batch_dft_to_audio(synth_dfts, listen=True, show_spectrogram=True, listen_max=2)
    
    mean_loss = loss_sum / n_batches
    
    return mean_loss