#!/usr/bin/env python
"""pop_embedding.py: Implementation of VAEs for the origins v44.3 dataset"""
__author__ = "Felix Pacheco"

# Basic libraries
import os
import numpy as np
import pickle
import pandas as pd

# Torch dependencies
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

# Custom methods
from src.data.pytorch_loading import SNPLoading
from src.features.preprocess_vae import one_hot_encoding, metadata_mapping, split_train_test, impute_data, get_enc_dict, loss_ignore_nans
from src.visualization.vae_out import de_encoding

##################################################
### Initialize hyper parameters, CUDA and seed ###
##################################################

# Hyperparams
CUDA = torch.cuda.is_available()
SEED = None  #Replace with your value
BATCH_SIZE = None  #Replace with your value 
EPOCHS = None  #Replace with your value
ZDIMS = None  #Replace with your value (Dimensions of latent space)
TRAIN = None  #Replace with your value (proportion of samples to keep on training set)
HIDDEN_UNITS = None  #Replace with your value (Units per layer)
HIDDEN_LAYERS = None  #Replace with your value (Amount of hidden layers)

# Set seed to GPU
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# Allow use of GPU memory
device = torch.device("cuda" if CUDA else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

#####################################
### Map metadata to observations  ###
#####################################

# Get files path
data_files_path = "/users/home/felpac/pop_cielab_2/data/v44_origins/bed_files/QC_origins_v44/DL_vae_v44.3/tensor_data/"
files = os.listdir("/users/home/felpac/pop_cielab_2/data/v44_origins/bed_files/QC_origins_v44/DL_vae_v44.3/tensor_data/")
files.sort()

# Map metadata to sample encodings/name to get labels
encodings_file = data_files_path+files[0]
metadata_path = "/users/home/felpac/pop_cielab_2/data/v44_origins/bed_files/QC_origins_v44/DL_vae_v44.3/metadata/v44_metadata_clear.tsv"
targets = metadata_mapping(encodings_file, metadata_path)

# Remove encoding file and variants file
X = files[1:-1] 
features = files[-1]

#########################################
### Encode targets : One-hot-encoding ###
#########################################

original_targets = np.array(targets)
targets = np.array(one_hot_encoding(targets))

# Make encoding dict to map encoding to original target
dict_encoding = get_enc_dict(original_targets, targets)
with open('/users/home/felpac/pop_cielab_2/data/v44_origins/bed_files/QC_origins_v44/DL_vae_v44.3/results/enc_dict', 'wb') as handle:
    pickle.dump(dict_encoding, handle)

####################################
### Partition train and test set ###
####################################

X_train, X_test, y_train, y_test = split_train_test(X, targets, 0.8)

train_set = SNPLoading(data_path=data_files_path, data_files=X_train, targets=y_train)
test_set = SNPLoading(data_path=data_files_path, data_files=X_test, targets=y_test)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

##################
### VAE Module ###
##################

class VAE(nn.Module):
    def __init__(self, input_features, input_batch, zdims,hidden_units, hidden_layers):
        super(VAE, self).__init__()
        
        # Input data
        self.input_features = input_features
        self.input_batch = input_batch
        self.zdims = zdims
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.relu = nn.ReLU()
        
        ### ENCODER : From input dimension to bottleneck (zdims)
        ## Input layer (fc1 : fully connected layer 1)
        #   Implement your code 

        ## Hidden layers (fcn)
        #   Implement your code

        ## Hidden to latent (fc3.1, fc3.2)
        #   Implement your code

        ### DECODER : From bottleneck to input dimension
        ## Latent to first hidden (fc4)
        #   Implement your code
         
        ## Hidden Layers (fcm)
        #   Implement your code

        ## Hidden to reconstructed input (fc5)
        #   Implement your code
        

    def encode(self, x, impute=True):
        """Input vector x -> fully connected layer 1 -> ReLU -> (fc21, fc22)
        Parameters
        ----------
        x : [input_batch, input_features] matrix

        Returns
        -------
        mu     : zdims mean units one for each latent dimension (fc21(h1))
        logvar :  zdims variance units one for each latent dimension (fc22(h1))
        """
        ## Input features -> hidden_units (fc1)
        #   Implement your code

        ## Hidden_units -> hidden units (fcn)
        #   Implement your code

        ## Hidden_units -> latent space (fc3.1,fc3.2, zdims)
        # Implement your code
        
        return mu, logvar

    def reparameterize(self, mu, logvar, inference=False):
        """Reparametrize to have variables instead of distribution functions
        Parameters
        ----------
        mu     : [input_batch, zdims] mean matrix
        logvar : [input_batch, zdims] variance matrix

        Returns
        -------
        During training random sample from the learned zdims-dimensional
        normal distribution; during inference its mean.
        """
        # Standard deviation
        std = torch.exp(0.5*logvar)
        # Noise term epsilon
        eps = torch.rand_like(std)
        
        if inference is True:
            return mu

        return mu+(eps*std)

    def decode(self, z):
        """z sample (20) -> fc3 -> ReLU (400) -> fc4 -> sigmoid -> reconstructed input
        Parameters
        ----------
        z : z vector

        Returns
        -------
        Reconstructed x'
        """
        ## zdims -> hidden
        #   Implement your code

        ## Hidden -> hidden
        #   Implement your code
        
        # Hidden -> input features
        #   Implement your code

        return reconstructed_input

    def forward(self, x):
        """Connects encoding and decoding by doing a forward pass"""
        # Get mu and logvar
        mu, logvar = self.encode(x)
        # Get latent samples
        z = self.reparameterize(mu, logvar)
        
        # Reconstruct input
        return self.decode(z), mu, logvar


# Get input features and encoding len of targets
input_features = train_set.__getitem__(1)[0].shape[0]
target_enc_len = targets.shape[1]

# Call model to device
model = VAE(input_features=input_features, input_batch=BATCH_SIZE, zdims=ZDIMS, hidden_units=HIDDEN_UNITS, hidden_layers=HIDDEN_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss function
def loss_function(recon_x, x, imputed_data, mu, logvar, input_features, input_batch, inference=False):
    """Computes the ELBO Loss (cross entropy + KLD)"""
    # KLD is Kullback–Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= (input_batch * zdims)

    # Compute loss between imputed x and reconstructed imputed x
    loss = nn.BCEWithLogitsLoss(reduction="none")
    BCE = loss(recon_x.flatten(), imputed_data.flatten()).flatten()

    # Compute BCE and ignore values that come from a nan
    BCE = loss_ignore_nans(BCE, x) ### This function is empty, you have to implement it
    BCE = torch.sum(BCE)/len(BCE)
    
    print(f"BCE : {BCE}, KLD : {KLD}")
    return BCE+ KLD, BCE, KLD


def train(epoch, model, train_loader, CUDA, optimizer, LOG_INTERVAL, input_features, input_batch):
    # toggle model to train mode
    model.train()
    train_loss = 0
    
    # Init save training
    train_loss_values = []
    train_bce = []
    train_kld = []

    # Iterate over train loader in batches of 20
    for batch_idx, (data, _) in enumerate(train_loader):
        
        if CUDA:
            data = data.to(device)
        
        optimizer.zero_grad()
        imputed_data = impute_data(tensor=data.cpu(), batch_size=input_batch ,categorical=True)  # Impute data function is also empty, you have to implement it

        if torch.cuda.is_available():
            data = data.to(device)
            imputed_data = imputed_data.to(device)


        # Push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(imputed_data)
        # calculate loss function
        loss, bce, kld = loss_function(recon_batch, data, imputed_data, mu, logvar, input_features, input_batch)
        
        # calculate the gradient of the loss w.r.t. the graph leaves
        loss.backward()
        train_loss += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        # Append values to then save them
        train_loss_values.append(loss.item())
        train_bce.append(bce.item())
        train_kld.append(kld.item())

        print(f"Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]")

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return train_loss_values, train_bce, train_kld


def test(epoch, model, test_loader, CUDA, optimizer, LOG_INTERVAL, input_features, input_batch, test_classes, zdims):
    # toggle model to test / inference mode
    test_loss = 0
    model.eval()

    # Save test loss
    test_loss_values = []
    test_bce = []
    test_kld = []

    # Save latent space
    mu_test = np.empty([0,zdims])
    targets_test = np.empty([0, test_classes])

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            #Impute data
            imputed_data = impute_data(tensor=data.cpu(), batch_size=input_batch, categorical=True)

            if CUDA:
                data = data.to(device)
                imputed_data = imputed_data.to(device)

            # Push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = model(imputed_data)
            
            # calculate loss function
            test_loss, bce, kld = loss_function(recon_batch, data, imputed_data, mu, logvar, input_features, input_batch)
            test_loss += test_loss.item()
            print(f"Test epoch: {epoch} [{i * len(data)}/{len(test_loader.dataset)}]")
            
            # Save test error 
            test_loss_values.append(test_loss.item())
            test_bce.append(bce.item())
            test_kld.append(kld.item())

            # Save latent space
            mu_ = mu.cpu().detach().numpy()
            target = _.cpu().detach().numpy()
            mu_test = np.append(mu_test, mu_, axis=0)
            targets_test = np.append(targets_test,target, axis=0)
            
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        return test_loss_values, test_bce, test_kld, mu_test, targets_test


train_loss_values = []
train_bce = []
train_kld = []

test_loss_values = []
test_bce = []
test_kld = []

epochs_save = [1, 5, 10, 20, 35, 50]

for epoch in range(1, EPOCHS + 1):
    train_loss_, train_bce_, train_kld_ = train(epoch, model, train_loader, CUDA, optimizer, LOG_INTERVAL, input_features, BATCH_SIZE)
    test_loss_, test_bce_, test_kld_, mu_test, targets_test = test(epoch, model, test_loader, CUDA, optimizer, LOG_INTERVAL, input_features, BATCH_SIZE, target_enc_len, ZDIMS)
    
    train_loss_values = train_loss_values + train_loss_
    train_bce = train_bce + train_bce_
    train_kld = train_kld + train_kld_

    test_loss_values = test_loss_values + test_loss_
    test_bce = test_bce + test_bce_
    test_kld = test_kld + test_kld_

    with open("../results/vae/train_loss.json", "wb") as fp:
        pickle.dump(train_loss_values, fp, protocol=3)

    with open("../results/vae/test_loss.json", "wb") as fp:
        pickle.dump(test_loss_values, fp, protocol=3)
    
    with open("../results/vae/train_bce.json", "wb") as fp:
        pickle.dump(train_bce, fp, protocol=3)

    with open("../results/vae/train_kld.json", "wb") as fp:
        pickle.dump(train_kld, fp, protocol=3)

    with open("../results/vae/test_bce.json", "wb") as fp:
        pickle.dump(test_bce, fp, protocol=3)

    with open("../results/vae/test.json", "wb") as fp:
        pickle.dump(test_kld, fp, protocol=3)

    # Save latent dimension every x epochs epochs
    if epoch in epochs_save:

        targets_test = de_encoding(targets_test, dict_encoding)
        df_test = pd.DataFrame({'label':targets_test, 'z1':mu_test[:,0], 'z2':mu_test[:,1]})
        df_test.to_csv("../results/latent_epoch"+str(epoch)+".csv")
        print(f"--> Epoch {epoch}: Saved latent values")