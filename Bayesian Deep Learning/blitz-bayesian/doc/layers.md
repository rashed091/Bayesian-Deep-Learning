# Bayesian Neural Network layers
They all inherit from torch.nn.Module
# Index:
  * [BayesianModule](#class-BayesianModule)
  * [BayesianLinear](#class-BayesianLinear)
  * [BayesianConv2d](#class-BayesianConv2d)
  * [BayesianLSTM](#class-BayesianLSTM)

---
## class BayesianModule(torch.nn.Module)
### blitz.modules.base_bayesian_module.BayesianModule()
Implements a as-interface used BayesianModule to enable further specific behavior
Inherits from torch.nn.Module

---

## class BayesianLinear
### blitz.modules.BayesianLinear(in_features, out_features, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)

Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper). 

Creates weight samplers of the class GaussianVariational for the weights and biases to be used on its feedforward ops.

Inherits from BayesianModule

#### Parameters:
  * in_features int -> Number nodes of the information to be feedforwarded
  * out_features int -> Number of out nodes of the layer
  * bias bool ->  wheter the model will have biases
  * prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
  * prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
  * freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
  
#### Methods:
  * forward():
      
      Performs a feedforward operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
      
   * forward_frozen(x):
      
      Performs a feedforward operation using onle the mu tensor as weights. 
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x = torch.tensor corresponding to the datapoints tensor to be feedforwarded
      
---
## class BayesianConv2d
### blitz.modules.BayesianConv2d(in_channels, out_channels, kernel_size, groups = 1, stride = 1, padding = 0, dilation = 1, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)
DESCRIPTION

#### Parameters:
  * in_channels int -> incoming channels for the layer
  * out_channels int -> output channels for the layer
  * kernel_size tuple (int, int) -> size of the kernels for this convolution layer
  * groups int -> number of groups on which the convolutions will happend
  * padding int -> size of padding (0 if no padding)
  * dilation int -> dilation of the weights applied on the input tensor
  * bias bool -> whether the bias will exist (True) or set to zero (False)
  * prior_sigma_1 float -> prior sigma on the mixture prior distribution 1
  * prior_sigma_2 float -> prior sigma on the mixture prior distribution 2
  * prior_pi float -> pi on the scaled mixture prior
  * freeze bool -> wheter the model will start with frozen(deterministic) weights, or not
  
#### Methods:
  * forward():
      
      Performs a feedforward Conv2d operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
      
   * forward_frozen(x):
      
      Performs a feedforward Conv2d operation using onle the mu tensor as weights. 
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x = torch.tensor corresponding to the datapoints tensor to be feedforwarded
    
---

## class BayesianLSTM
### blitz.modules.BayesianLSTM(in_features, out_features, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)

Bayesian LSTM layer, implements the LSTM layer using the weight uncertainty tools proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper). 

Creates weight samplers of the class GaussianVariational for the weights and biases to be used on its feedforward ops.

Inherits from BayesianModule

#### Parameters:
  * in_features int -> Number nodes of the information to be feedforwarded
  * out_features int -> Number of out nodes of the layer
  * bias bool ->  wheter the model will have biases
  * prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
  * prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
  * freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
  
#### Methods:
  * forward(x, ):
      
      Performs a feedforward operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns  tuple of format (torch.tensor, (torch.tensor, torch.tensor)), representing the output and hidden state of the LSTM layer
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
       * hidden_states - None or tupl of the format (torch.tensor, torch.tensor), representing the hidden states of the network. Internally, if None, consider zeros of the proper format).
      
   * sample_weights():
      
      Assings internally its weights to be used on feedforward operations by sampling it from its GaussianVariational
      
   * get_frozen_weights():
   
      Assings internally for its weights deterministaclly the mean of its GaussianVariational sampler.
      
