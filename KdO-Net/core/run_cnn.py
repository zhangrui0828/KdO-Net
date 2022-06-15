
from config.config import parser
from core.cnn import network



def to_network(net_args):
    config_arguments = parser.parse_args(net_args)

    kdo_net = network.NetworkBuilder(config_arguments)
    print('Run mode "{}" selected.'.format(config_arguments.run_mode))
    # Select the run mode
    print('config_arguments-1:',config_arguments)
    if config_arguments.run_mode == "train":
        for num_epoch in range(config_arguments.max_epochs):
            if num_epoch==0:
                 kdo_net.train(num_epoch)
            else:
                config_arguments.resume_flag=1
                config_arguments.pretrained_model='../models/32_dim/'+'lr_{}_batchSize_{}_outDim_{}_epoch_{}_Iteration_{}.ckpt'.format(
                                                    config_arguments.learning_rate,
                                                    config_arguments.batch_size,
                                                    config_arguments.output_dim,
                                                    0, config_arguments.max_steps-1)
                print('config_arguments-2:',config_arguments)
                kdo_net.train(num_epoch)


    elif config_arguments.run_mode == "test":
        # Evaluate the network
        kdo_net.test()
    else:
        raise ValueError('%s is not a valid run mode.'.format(config_arguments.run_mode))
    # config_arguments, unparsed_arguments = parser.parse_args(["--decay_rate=0.99","--decay_step=6666"])

