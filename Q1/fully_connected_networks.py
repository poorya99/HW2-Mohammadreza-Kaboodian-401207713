"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
from libs import Solver


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in out.  #
        # You will need to reshape the input into rows.                      #
        ######################################################################
        # Replace "pass" statement with your code
        N = x.shape[0]
        # print(x.shape)
        # print(w.shape)
        # print(b.shape)
        out = torch.matmul(x.reshape(N,-1), w)+b
        # print(out.shape)

        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # TODO: Implement the linear backward pass.      #
        ##################################################
        # Replace "pass" statement with your code
        x.requires_grad = True
        w.requires_grad = True
        b.requires_grad = True
        # dout.requires_grad = True
        N = x.shape[0]

        out = torch.matmul(x.reshape(N, -1), w) + b
        # print(dout.shape, out.shape)
        (out * dout).sum().backward()
        # print(w.grad)
        dx, dw, db = (x.grad, w.grad, b.grad)
        dx.detach_()
        dw.detach_()
        db.detach_()
        ##################################################
        #                END OF YOUR CODE                #
        ##################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor with an  #
        # in-place operation.                             #
        ###################################################
        # Replace "pass" statement with your code
        out = torch.max(torch.tensor([0],device="cuda"), x)
        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # TODO: Implement the ReLU backward pass.           #
        # You should not change the input tensor with an    #
        # in-place operation.                               #
        #####################################################
        # Replace "pass" statement with your code
        x.requires_grad = True
        out = torch.maximum(torch.tensor([0],device="cuda"), x)
        # print(out.shape, dout.shape)
        (dout * out).sum().backward()
        dx = x.grad
        dx.detach_()
        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass (hint: cache = (fc_cache, relu_cache))
        """
        out = None
        cache = None
        ######################################################################
        # TODO: Implement the linear-relu forward pass.                      #
        ######################################################################
        # Replace "pass" statement with your code
        N = x.shape[0]
        # print(x.device, w.device,b.device)
        if x.device != "cuda":
          x=x.to("cuda")
        # print(x.device, w.device,b.device)
        out = torch.matmul(x.reshape(N, -1), w) + b
        out = torch.max(torch.tensor([0.0],device="cuda"), out)
        cache = (x, w, b)
        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the linear-relu backward pass.                     #
        ######################################################################
        # Replace "pass" statement with your code
        (x, w, b) = cache
        N = x.shape[0]
        X = x.clone(); W = w.clone(); B = b.clone();
        X.requires_grad = True
        W.requires_grad = True
        B.requires_grad = True
        out = torch.matmul(X.reshape(N, -1), W) + B
        out = torch.max(torch.tensor([0],device="cuda"), out)
        (out * dout).sum().backward()
        (dx, dw, db) = (X.grad, W.grad, B.grad)
        dx.detach_()
        dw.detach_()
        db.detach_()
        ######################################################################
        #                END OF YOUR CODE                                    #
        ######################################################################
        return dx, dw, db

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss = None
    dx = None
    ######################################################################
    # TODO: Implement the Softmax layer.                                 #
    ######################################################################
    # Replace "pass" statement with your code
    # print(x.shape)
    max_regulizer, _ = torch.max(x, axis=1, keepdims=True)
    # print(tmp.shape)

    regularized_logits = x - max_regulizer
    Z = torch.sum(torch.exp(regularized_logits), axis=1, keepdims=True)
    log_probs = regularized_logits - torch.log(Z)
    probs = torch.exp(log_probs)
    N = x.shape[0]
    loss = -torch.sum(log_probs[torch.arange(N), y]) / N
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    # if x.requires_grad == False:
    #   X = x.clone()
    #   X.requires_grad = True
    #   max_regulizer_tmp, _ = torch.max(X, axis=1, keepdims=True)
    #   # print(tmp.shape)
    #   regularized_logits_tmp = X - max_regulizer_tmp
    #   Z_tmp = torch.sum(torch.exp(regularized_logits_tmp), axis=1, keepdims=True)
    #   log_probs_tmp = regularized_logits_tmp - torch.log(Z_tmp)
    #   probs_tmp = torch.exp(log_probs_tmp)
    #   loss_tmp = -torch.sum(log_probs_tmp[torch.arange(N), y]) / N

    #   loss_tmp.backward()
    #   dx = X.grad
    # else:
    #   loss.backward()
    #   dx = x.grad

    # print(loss.item())
    # X.grad.zero_()
    ######################################################################
    #                END OF YOUR CODE                                    #
    ######################################################################
    return loss, dx



class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: Initialize the weights and biases of the two-layer net.   #
        # Weights should be initialized from a Gaussian centered at       #
        # 0.0 with standard deviation equal to weight_scale, and biases   #
        # should be initialized to zero. All weights and biases should    #
        # be stored in the dictionary self.params, with first layer       #
        # weights and biases using the keys 'W1' and 'b1' and second layer#
        # weights and biases using the keys 'W2' and 'b2'.                #
        ###################################################################
        # Replace "pass" statement with your code
        self.params["W1"] = torch.randn((input_dim, hidden_dim),device="cuda",dtype=dtype) * weight_scale
        self.params["b1"] = torch.zeros(hidden_dim,device="cuda",dtype=dtype)
        self.params["W2"] = torch.randn((hidden_dim, num_classes),device="cuda",dtype=dtype) * weight_scale
        self.params["b2"] = torch.zeros(num_classes,device="cuda",dtype=dtype)
        ###############################################################
        #                            END OF YOUR CODE                 #
        ###############################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################
        # Replace "pass" statement with your code
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        lin_relu1 = Linear_ReLU()
        lin_relu2 = Linear_ReLU()

        out, cache1 = lin_relu1.forward(X, W1, b1)

        out, cache2 = lin_relu2.forward(out, W2, b2)
        scores = out
        # softmax_loss(out, y)
        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the two-layer net.        #
        # Store the loss in the loss variable and gradients in the grads  #
        # dictionary. Compute data loss using softmax, and make sure that #
        # grads[k] holds the gradients for self.params[k]. Don't forget   #
        # to add L2 regularization!                                       #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and       #
        # you pass the automated tests, make sure that your L2            #
        # regularization does not include a factor of 0.5.                #
        ###################################################################
        # Replace "pass" statement with your code

        softmax_loss0, dscore = softmax_loss(out, y)
        regularization_term =  self.reg * (torch.sum(W1**2) + torch.sum(W2**2))
        loss = softmax_loss0 + regularization_term

        dx2, dw2, db2 = lin_relu2.backward(dscore, cache2)
        dx1, dw1, db1 = lin_relu1.backward(dx2, cache1)

        grads["b1"] = db1.detach()
        grads["b2"] = db2.detach()
        grads["W2"] = dw2.detach() + 2*self.reg * W2.detach()
        grads["W1"] = dw1.detach() + 2*self.reg * W1.detach()
        loss=loss.detach()
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

        return loss, grads




class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all         #
        # values in the self.params dictionary. Store weights and biases      #
        # for the first layer in W1 and b1; for the second layer use W2 and   #
        # b2, etc. Weights should be initialized from a normal distribution   #
        # centered at 0 with standard deviation equal to weight_scale. Biases #
        # should be initialized to zero.                                      #
        #######################################################################
        # Replace "pass" statement with your code
        self.hidden_dims = hidden_dims
        for i,hidden_dim in enumerate(hidden_dims):
          dict_key_w = "W" + str(i+1)
          dict_key_b = "b" + str(i+1)
          # print(dict_key_w,dict_key_b,)
          if i == 0:
            self.params[dict_key_w] = torch.randn((input_dim, hidden_dims[i]),device="cuda",dtype=dtype) * weight_scale
            self.params[dict_key_b] = torch.zeros(hidden_dims[i],device="cuda",dtype=dtype)
          else:
            self.params[dict_key_w] = torch.randn((hidden_dims[i-1], hidden_dims[i]),device="cuda",dtype=dtype) * weight_scale
            self.params[dict_key_b] = torch.zeros(hidden_dims[i],device="cuda",dtype=dtype)
        dict_key_w = "W" + str(i+2)
        dict_key_b = "b" + str(i+2)
        self.params[dict_key_w] = torch.randn((hidden_dims[i], num_classes),device="cuda",dtype=dtype) * weight_scale
        self.params[dict_key_b] = torch.zeros(num_classes,device="cuda",dtype=dtype)
        # for s in self.params:
        #   print(s,self.params[s].shape)
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ##################################################################
        # TODO: Implement the forward pass for the fully-connected net,  #
        # computing the class scores for X and storing them in the       #
        # scores variable.                                               #
        #                                                                #
        # When using dropout, you'll need to pass self.dropout_param     #
        # to each dropout forward pass.                                  #
        ##################################################################
        # Replace "pass" statement with your code

        layers_activation_func = []
        caches = []

        for i,hidden_dim in enumerate(self.hidden_dims):
          dict_key_w = "W" + str(i+1)
          dict_key_b = "b" + str(i+1)
          layers_activation_func.append(Linear_ReLU())
          if(i ==0):
            out, cache = layers_activation_func[i].forward(X, self.params[dict_key_w],
                        self.params[dict_key_b])
            caches.append(cache)
          else:
            out, cache = layers_activation_func[i].forward(out, self.params[dict_key_w],
                        self.params[dict_key_b])
            caches.append(cache)
        dict_key_w = "W" + str(i+2)
        dict_key_b = "b" + str(i+2)
        layers_activation_func.append(Linear_ReLU())
        out, cache = layers_activation_func[i+1].forward(out, self.params[dict_key_w],
                        self.params[dict_key_b])
        caches.append(cache)
        scores = out
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #####################################################################
        # TODO: Implement the backward pass for the fully-connected net.    #
        # Store the loss in the loss variable and gradients in the grads    #
        # dictionary. Compute data loss using softmax, and make sure that   #
        # grads[k] holds the gradients for self.params[k]. Don't forget to  #
        # add L2 regularization!                                            #
        # NOTE: To ensure that your implementation matches ours and you     #
        # pass the automated tests, make sure that your L2 regularization   #
        # includes a factor of 0.5 to simplify the expression for           #
        # the gradient.                                                     #
        #####################################################################
        # Replace "pass" statement with your code
        softmax_loss0, dscore = softmax_loss(out, y)

        regularization_term = 0
        num_hidden_layers = len(self.hidden_dims)
        dx, dw, db = self.num_layers*[None], self.num_layers*[None], self.num_layers*[None]
        for i in range(num_hidden_layers+1):
          idx = num_hidden_layers-i
          dict_key_w = "W" + str(idx+1)
          dict_key_b = "b" + str(idx+1)
          # print("***", dict_key_w, "i=",i, "idx =",idx  )
          # a,b,c = caches[idx]
          # print(dscore.shape, a.shape, b.shape, c.shape)
          if i==0:
            dx_t, dw_t, db_t = layers_activation_func[idx].backward(dscore, caches[idx])
          else:
            dx_t, dw_t, db_t = layers_activation_func[idx].backward(dx_t, caches[idx])
          
          dx[idx], dw[idx], db[idx] = dx_t, dw_t, db_t
          grads[dict_key_b] = db_t.detach()
          grads[dict_key_w] = dw_t.detach() + 2*self.reg * self.params[dict_key_w].detach()

          regularization_term += torch.sum(self.params[dict_key_w]**2)
        
        regularization_term = self.reg * regularization_term
        loss = softmax_loss0 + regularization_term
        loss=loss.detach()
        # for s in grads:
        #   print(s)
        ###########################################################
        #                   END OF YOUR CODE                      #
        ###########################################################

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    #############################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that   #
    # achieves at least 50% accuracy on the validation set.     #
    #############################################################
    solver = None
    # Replace "pass" statement with your code

    solver = Solver(model=model, data=data_dict,device=device,
                    optim_config={'learning_rate':0.007},
                     num_epochs=90)
    ##############################################################
    #                    END OF YOUR CODE                        #
    ##############################################################
    return solver


def get_three_layer_network_params():
    ###############################################################
    # TODO: Change weight_scale and learning_rate so your         #
    # model achieves 100% training accuracy within 20 epochs.     #
    ###############################################################
    weight_scale = 1e-2   # Experiment with this!
    learning_rate = 1e-4  # Experiment with this!
    ################################################################
    #                             END OF YOUR CODE                 #
    ################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ################################################################
    # TODO: Change weight_scale and learning_rate so your          #
    # model achieves 100% training accuracy within 20 epochs.      #
    ################################################################
    learning_rate = 2e-3  # Experiment with this!
    weight_scale = 1e-5   # Experiment with this!
    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))
    
    next_w = None
    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    # Replace "pass" statement with your code
    v = config['momentum']*v-config['learning_rate'] * dw
    next_w =w+v
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # Replace "pass" statement with your code

    cache = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw**2
    next_w = w - config['learning_rate'] * dw / (torch.sqrt(cache) + config['epsilon'])
    config['cache'] = cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ##########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in#
    # the next_w variable. Don't forget to update the m, v, and t variables  #
    # stored in config.                                                      #
    #                                                                        #
    # NOTE: In order to match the reference output, please modify t _before_ #
    # using it in any calculations.                                          #
    ##########################################################################
    # Replace "pass" statement with your code
    beta1 = config['beta1']
    beta2 = config['beta2']
    m = config['m']
    t = config['t'] +1
    v = config['v']
    eps = config['epsilon'] 
    learning_rate = config['learning_rate']
    m = beta1*m + (1-beta1)*dw
    mt = m / (1-beta1**t)
    v = beta2*v + (1-beta2)*(dw**2)
    vt = v / (1-beta2**t)
    next_w = w - learning_rate * mt / (torch.sqrt(vt) + eps)
    # m = beta1*m + (1-beta1)*dw
    # v = beta2*v + (1-beta2)*(dw**2)
    # next_w = w - learning_rate * m / (torch.sqrt(v) + eps)
    config['v'] = v
    config['m'] = m
    config['t'] = t
    #########################################################################
    #                              END OF YOUR CODE                         #
    #########################################################################

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            # Replace "pass" statement with your code
            mask = ( torch.rand(x.shape,device='cuda') < p ) / p
            out = x * mask
            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == 'test':
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            # Replace "pass" statement with your code
            out = x
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################
            # TODO: Implement training phase backward pass for        #
            # inverted dropout                                        #
            ###########################################################
            # Replace "pass" statement with your code
            dx = dout * mask
            ###########################################################
            #                     END OF YOUR CODE                    #
            ###########################################################
        elif mode == 'test':
            dx = dout
        return dx

