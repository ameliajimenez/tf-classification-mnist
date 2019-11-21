import os
import numpy as np
from data_loader import read_data_sets
import convnet

# working directory
workingdir = os.getcwd()

# models directory
modeldir = './models/example/'

# optimization parameters
nepochs = 15  # 15
name_opt = 'adam'
momentum = 0.9
lr = 1e-3  # learning rate
decay_rate = 0.1  # decay learning rate by x
decay_after_epoch = 10  # decay learning rate after x epochs
batch_size = 128
dropout = 0.9
cost = 'cross_entropy'  # loss to minimize

# load data
perc = 30  # percentage of training data
datadir = os.path.join(os.getcwd(), './data/mnist')  # data directory
data_provider = read_data_sets(datadir, percentage_train=perc/100.0)
n_train = data_provider.train.num_examples
print('Number of training images {:d}'.format(n_train))
# more training parameters
iters_per_epoch = np.ceil(1.0 * n_train / batch_size).astype(np.int32)
decay_steps = decay_after_epoch * iters_per_epoch
opt_kwargs = dict(learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)

# definition of the network
net = convnet.ConvNet(channels=1, n_class=10, is_training=True, cost_name=cost)

# definition of the trainer
trainer = convnet.Trainer(net, optimizer=name_opt, batch_size=batch_size, opt_kwargs=opt_kwargs)

# start training
path = trainer.train(data_provider, modeldir, training_iters=iters_per_epoch, epochs=nepochs, dropout=dropout)

print('Optimization Finished!')