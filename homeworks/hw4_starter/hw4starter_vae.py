from __future__ import print_function
import argparse
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            q_sigma=0.2,
            n_dims_code=2,
            n_dims_data=784,
            hidden_layer_sizes=[32]):
        super(VariationalAutoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.q_sigma = torch.Tensor([float(q_sigma)])
        encoder_layer_sizes = (
            [n_dims_data] + hidden_layer_sizes + [n_dims_code]
            )
        self.n_layers = len(encoder_layer_sizes) - 1
        # Create the encoder, layer by layer
        self.encoder_activations = list()
        self.encoder_params = nn.ModuleList()
        for layer_id, (n_in, n_out) in enumerate(zip(
                encoder_layer_sizes[:-1], encoder_layer_sizes[1:])):
            self.encoder_params.append(nn.Linear(n_in, n_out))
            self.encoder_activations.append(F.relu)
        self.encoder_activations[-1] = lambda a: a

        self.decoder_activations = list()
        self.decoder_params = nn.ModuleList()
        decoder_layer_sizes = [a for a in reversed(encoder_layer_sizes)]
        for (n_in, n_out) in zip(
                decoder_layer_sizes[:-1], decoder_layer_sizes[1:]):
            self.decoder_params.append(nn.Linear(n_in, n_out))
            self.decoder_activations.append(F.relu)
        self.decoder_activations[-1] = torch.sigmoid

    def forward(self, x_ND):
        """ Run entire probabilistic autoencoder on input (encode then decode)

        Returns
        -------
        xproba_ND : 1D array, size of x_ND
        """
        mu_NC = self.encode(x_ND)
        z_NC = self.draw_sample_from_q(mu_NC)
        return self.decode(z_NC), mu_NC

    def draw_sample_from_q(self, mu_NC):
        ''' Draw sample from the probabilistic encoder q(z|mu(x), \sigma)

        We assume that "q" is Normal with:
        * mean mu (argument of this function)
        * stddev q_sigma (attribute of this class, use self.q_sigma)

        Args
        ----
        mu_NC : tensor-like, N x C
            Mean of the encoding for each of the N images in minibatch.

        Returns
        -------
        z_NC : tensor-like, N x C
            Exactly one sample vector for each of the N images in minibatch.
        '''
        N = mu_NC.shape[0]
        C = self.n_dims_code
        if self.training:
            # Draw standard normal samples "epsilon"
            eps_NC = torch.randn(N, C)
            ## TODO
            # Using reparameterization trick,
            # Write a procedure here to make z_NC a valid draw from q 
            z_NC = None # <-- TODO fix me
            return z_NC
        else:
            # For evaluations, we always just use the mean
            return mu_NC


    def encode(self, x_ND):
        cur_arr = x_ND
        for ll in range(self.n_layers):
            linear_func = self.encoder_params[ll]
            a_func = self.encoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        mu_NC = cur_arr
        return mu_NC

    def decode(self, z_NC):
        cur_arr = z_NC
        for ll in range(self.n_layers):
            linear_func = self.decoder_params[ll]
            a_func = self.decoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        xproba_ND = cur_arr
        return xproba_ND

    def calc_vi_loss(self, x_ND, n_mc_samples=1):
        total_loss = 0.0
        mu_NC = self.encode(x_ND)
        for ss in range(n_mc_samples):
            sample_z_NC = self.draw_sample_from_q(mu_NC)
            sample_xproba_ND = self.decode(sample_z_NC)
            sample_bce_loss = F.binary_cross_entropy(
                sample_xproba_ND, x_ND, reduction='sum')

            # KL divergence from q(mu, sigma) to prior (std normal)
            # see Appendix B from VAE paper
            # https://arxiv.org/pdf/1312.6114.pdf
            kl = 0.0 # <- TODO fix me
            total_loss += sample_bce_loss + kl
        return total_loss / float(n_mc_samples), sample_xproba_ND


def train_for_one_epoch_of_gradient_update_steps(
        model, optimizer, train_loader, epoch, args):
    model.train()
    train_loss = 0.0
    n_seen = 0
    for batch_idx, (batch_data, _) in enumerate(train_loader):
        # Reshape the data from n_images x 28x28 to n_images x 784 (NxD)
        batch_x_ND = batch_data.to(device).view(-1, model.n_dims_data)

        # Zero out any stored gradients attached to the optimizer
        optimizer.zero_grad()

        # Compute the loss (and the required reconstruction as well)
        loss, batch_xproba_ND = model.calc_vi_loss(
            batch_x_ND, n_mc_samples=args.n_mc_samples)

        # Increment the total loss (over all batches)
        train_loss += loss.item()

        # Compute the gradient of the loss wrt model parameters
        # (gradients are stored as attributes of parameters of 'model')
        loss.backward()

        # Take an optimization step (gradient descent step)
        optimizer.step()

        n_seen += batch_x_ND.shape[0]
        if (1+batch_idx) % (len(train_loader)//10)  == 0:
            l1_dist = torch.mean(torch.abs(batch_x_ND - batch_xproba_ND))
            print("  epoch %3d | frac_seen %.3f | avg loss %.3e | batch loss % .3e | batch l1 % .3f" % (
                epoch, (1+batch_idx) / float(len(train_loader)),
                train_loss / float(n_seen),
                loss.item() / float(batch_x_ND.shape[0]),
                l1_dist,
                ))
    return model


def eval_model_on_data(
        model, data_loader, device, args):
    model.eval()
    total_vi_loss = 0.0
    total_l1 = 0.0
    total_bce = 0.0
    n_seen = 0
    total_1pix = 0.0
    for batch_idx, (batch_data, _) in enumerate(data_loader):
        batch_x_ND = batch_data.to(device).view(-1, model.n_dims_data)
        total_1pix += torch.sum(batch_x_ND)
        loss, _ = model.calc_vi_loss(batch_x_ND, n_mc_samples=args.n_mc_samples)
        total_vi_loss += loss.item()

        # Use deterministic reconstruction to evaluate bce and l1 terms
        batch_xproba_ND = model.decode(model.encode(batch_x_ND))
        total_l1 += torch.sum(torch.abs(batch_x_ND - batch_xproba_ND))
        total_bce += F.binary_cross_entropy(batch_xproba_ND, batch_x_ND, reduction='sum')
        n_seen += batch_x_ND.shape[0]
    print("Total images %d. Total on pixels: %d. Frac pixels on: %.3f" % (
        n_seen, total_1pix, total_1pix / float(n_seen*784)))

    vi_loss_per_pixel = total_vi_loss / float(n_seen * model.n_dims_data)
    l1_per_pixel = total_l1 / float(n_seen * model.n_dims_data)
    bce_per_pixel = total_bce / float(n_seen * model.n_dims_data) 
    return float(vi_loss_per_pixel), float(l1_per_pixel), float(bce_per_pixel)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder MNIST Example')
    parser.add_argument(
        '--n_epochs', type=int, default=10,
        help="number of epochs (default: 10)")
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size (default: 1024)')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for grad. descent (default: 0.001)')
    parser.add_argument(
        '--hidden_layer_sizes', type=str, default='32',
        help='Comma-separated list of size values (default: "32")')
    parser.add_argument(
        '--filename_prefix', type=str, default='AE-arch=$hidden_layer_sizes-lr=$lr')
    parser.add_argument(
        '--q_sigma', type=float, default=0.1,
        help='Fixed variance of approximate posterior (default: 0.1)')
    parser.add_argument(
       '--n_mc_samples', type=int, default=1,
       help='Number of Monte Carlo samples (default: 1)')
    parser.add_argument(  
        '--seed', type=int, default=8675309,
        help='random seed (default: 8675309)')
    args = parser.parse_args()
    args.hidden_layer_sizes = [int(s) for s in args.hidden_layer_sizes.split(',')]

    ## Set random seed
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    ## Set filename_prefix for results
    for key, val in args.__dict__.items():
        args.filename_prefix = args.filename_prefix.replace('$' + key, str(val))
    print("Saving with prefix: %s" % args.filename_prefix)

    ## Create generators for grabbing batches of train or test data
    # Each loader will produce **binary** data arrays (using transforms defined below)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), torch.round])),    
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=False,
            transform=transforms.Compose([transforms.ToTensor(), torch.round])),
        batch_size=args.batch_size, shuffle=True)

    ## Create VAE model by calling its constructor
    model = VariationalAutoencoder(
        q_sigma=args.q_sigma,
        hidden_layer_sizes=args.hidden_layer_sizes)
    model = model.to(device)

    ## Create an optimizer linked to the model parameters
    # Given gradients computed by pytorch, this optimizer handle update steps to params
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ## Training loop that repeats for each epoch:
    #  -- perform minibatch training updates (one epoch = one pass thru dataset)
    #  -- for latest model, compute performance metrics on training set
    #  -- for latest model, compute performance metrics on test set
    for epoch in range(args.n_epochs + 1):
        if epoch > 0:
            model = train_for_one_epoch_of_gradient_update_steps(
                model, optimizer, train_loader, epoch, args)

        ## Only save results for epochs 0,1,2,3,4,5 and 10,20,30,...
        if epoch > 5 and epoch % 10 != 0:
            continue

        ## Compute VI loss (bce + kl), bce alone, and l1 alone
        tr_loss, tr_l1, tr_bce = eval_model_on_data(
            model, train_loader, device, args)
        print('  epoch %3d  train loss %.3f  bce %.3f  l1 %.3f' % (epoch, tr_loss, tr_bce, tr_l1))
        te_loss, te_l1, te_bce = eval_model_on_data(
            model, test_loader, device, args)
        print('  epoch %3d  test  loss %.3f  bce %.3f  l1 %.3f' % (epoch, te_loss, te_bce, te_l1))

        ## Write perf metrics to CSV file (so we can easily plot later)
        #
        # Create str repr of architecture size list: [20,30] becomes '[20;30]'
        arch_str = '[' + ';'.join(map(str,args.hidden_layer_sizes)) + ']'
        row_df = pd.DataFrame([[
                epoch, tr_loss, tr_l1, tr_bce, te_loss, te_l1, te_bce,
                arch_str, args.lr, args.q_sigma, args.n_mc_samples]],
            columns=[
                'epoch', 'tr_vi_loss', 'tr_l1_error', 'tr_bce_error', 'te_vi_loss', 'te_l1_error', 'te_bce_error',
                'arch_str', 'lr', 'q_sigma', 'n_mc_samples'])
        csv_str = row_df.to_csv(
            None,
            float_format='%.8f',
            index=False,
            header=False if epoch > 0 else True,
            )
        if epoch == 0:
            # At start, write to a clean file with mode 'w'
            with open('%s_perf_metrics.csv' % args.filename_prefix, 'w') as f:
                f.write(csv_str)
        else:
            # Append to existing file with mode 'a'
            with open('%s_perf_metrics.csv' % args.filename_prefix, 'a') as f:
                f.write(csv_str)

        ## Make pretty plots of random samples in code space decoding into data space
        with torch.no_grad():
            P = int(np.sqrt(model.n_dims_data))
            sample = torch.randn(25, model.n_dims_code).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(25, 1, P, P), nrow=5, padding=4,
                       filename='%s_sample_images_epoch=%03d.png' % (args.filename_prefix, epoch))


