import pathlib
import torch
import numpy as np
import json
from src.models.v1.model import AudioAutoEncoder


def get_model(model_timestamp, 
              model_name,
              models_path, 
              epoch = -1):
    autoencoder = AudioAutoEncoder(2, 32, 400)
    
    if epoch == -1:
        pass
    
    else:
        model_checkpoint_basename = models_path / f"{model_name}__epoch_{epoch}"
        model_checkpoint_epoch = pathlib.Path(f"{model_checkpoint_basename}.ckpt")
        model_loss_filename = pathlib.Path(f"{model_checkpoint_basename}_loss.npy")
        model_loss_test_filename = pathlib.Path(f"{model_checkpoint_basename}_loss_testing.npy")
        model_hyperparams_filename = models_path / model_timestamp / "hyperparams.json"

    with open(model_hyperparams_filename, "r") as f:
        model_hyperparams_dict = json.load(f)

    assert model_checkpoint_epoch.exists()
    assert model_loss_filename.exists()
    assert model_loss_test_filename.exists()
    assert model_hyperparams_filename.exists()
    
    print(f"Opening model {model_checkpoint_epoch}...")

    trained_model_dict = torch.load(model_checkpoint_epoch)

    autoencoder.load_state_dict(trained_model_dict)
    
    print(f"Loaded model {model_checkpoint_epoch}")
    
    losses_list = list(np.load(model_loss_filename))
    losses_list_testing = list(np.load(model_loss_test_filename))
    

    #Initialize indepdent optimizer for both networks
    learning_rate = model_hyperparams_dict["learning_rate"]
    weight_decay = model_hyperparams_dict["weight_decay"]
    optimizer = torch.optim.Adam(autoencoder.parameters(),
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    lbmda = model_hyperparams_dict["lambda"]

    num_epochs = model_hyperparams_dict["target_num_epochs"]
    
    return [autoencoder, 
            learning_rate, 
            weight_decay, 
            optimizer, 
            lbmda,
            num_epochs,
            losses_list,
            losses_list_testing]
    