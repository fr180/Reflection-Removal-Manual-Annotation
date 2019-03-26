import argparse

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default = "./datasets/train", help='reflect')
parser.add_argument('--val_dataset', type=str,default = "./datasets/test", help='reflect')

parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input images to network')
parser.add_argument('--epoch', type=int, default=100, help='the iters of training')

parser.add_argument('--image_num', type=int, default=13500, help='the number of sample')
parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--d_lr', type=float, default=0.00001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--gpu', type=str,default="1", help='enables cuda')

parser.add_argument('--gen_weight', type=str,default="./weights/generator.h5", help="path to netG (to continue training)")
parser.add_argument('--dis_weight', type=str,default="./weights/discriminator.h5", help="path to netD (to continue training)")

config = parser.parse_args()
