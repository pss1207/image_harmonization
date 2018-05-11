import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Model(BaseModel):
    def name(self):
        return 'Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['loss']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['input_image', 'input_mask', 'target_image', 'prediction']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['Model']

        # load/define networks
        # The naming conversion is different from those used in the paper
        self.net = networks.define(opt.input_nc, opt.output_nc, opt.init_type, self.gpu_ids)


        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.schedulers = []
            self.schedulers.append(networks.get_scheduler(self.optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        input_image = input[0]
        input_mask = input[1]
        target_image = input[2]
        input_array = torch.cat((input_image, input_mask), dim=1)
        if len(self.gpu_ids) > 0:
            input_array = input_array.cuda(self.gpu_ids[0], async=True)
            target_image = target_image.cuda(self.gpu_ids[0], async=True)

        self.input_array = input_array
        self.target_image = target_image


    def forward(self):
        self.input_array = Variable(self.input_array)
        self.prediction = self.net(self.input_array)
        self.target_image = Variable(self.target_image)
        self.input_image = self.input_array[:,:3, :, :]
        self.input_mask = self.input_array[:, 3, :, :]

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.input_array = Variable(self.input_array)
            self.prediction = self.net(self.input_array)
            self.target_image = Variable(self.target_image)
            self.input_image = self.input_array[:, :3, :, :]
            self.input_mask = self.input_array[:, 3, :, :]


    def backward(self):
        self.loss = self.criterion(self.prediction, self.target_image)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

