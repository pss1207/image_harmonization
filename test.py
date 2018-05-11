import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Phase = %s, Epoch = %s' % (opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        print('%04d: process image...' % (i))
        visualizer.save_images(webpage, visuals, i, aspect_ratio=opt.aspect_ratio)

    webpage.save()
