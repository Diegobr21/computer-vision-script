import wget
from os import mkdir, path
from os.path import join, abspath, dirname, exists

weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
names_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
config_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'

file_path = abspath(__file__)
file_parent_dir = dirname(file_path)
config_dir = join(file_parent_dir, 'config')
inputs_dir = join(file_parent_dir, 'inputs')

if not exists(config_dir):
    mkdir(config_dir)
    wget.download(weights_url, out=join(config_dir, 'yolov3.weights'))
    wget.download(names_url, out=join(config_dir, 'coco.names'))
    wget.download(config_url, out=join(config_dir, 'yolov3.cfg'))

if not exists(inputs_dir):
    mkdir(inputs_dir)

# YOU NEED TO RUN THIS IN ORDER TO GET THE NECESSARY CONFIG FILES FOR THE NEURAL NET