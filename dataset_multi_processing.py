import numpy as np
import os
import cv2
import scipy.stats as st
from cfg import config, IMG_EXTENSIONS


class DatasetProcessing():
    def __init__(self,batch_size,pic_size):

        self.data_syn_dir = config.dataset+'/synthetic/'
        self.data_real_dir = config.dataset+'/real/'


        self.id = 0
        self.count = 0
        self.batch_size = batch_size
        self.pic_size = pic_size
        self.k_sz = np.linspace(1, 5, 80)  # for synthetic images
        # todo: images pairs for generating synthetic training images
        self._, self.syn_image1_list, self.syn_image2_list = self.prepare_train_data(self.data_syn_dir)
        # todo: no reflection ground truth for real images
        self.input_real_names, self.output_real_names1, self.output_real_names2 = self.prepare_train_data(self.data_real_dir)

        self.num_train = len(self.syn_image1_list) + len(self.output_real_names1)
        self.all_l = np.zeros(self.num_train, dtype=float)
        self.all_percep = np.zeros(self.num_train, dtype=float)
        self.all_grad = np.zeros(self.num_train, dtype=float)
        self.all_g = np.zeros(self.num_train, dtype=float)

        # self.input_images = np.zeros([self.batch_size,self.pic_size,self.pic_size,3])
        # self.output_images = np.zeros([self.batch_size,self.pic_size,self.pic_size,3])
        self.input_images = []
        self.output_images = []

        self.ids = self.id_random()

        # create a vignetting mask
        self.g_mask = self.gkern(560, 3)
        self.g_mask = np.dstack((self.g_mask, self.g_mask, self.g_mask))

    def __next__(self):

        self.id = self.ids[self.count:(self.count+self.batch_size)]
        self.count += self.batch_size
        return self.id

    def id_random(self):
        id_num = []
        for id in np.random.permutation(self.num_train):
            id_num.append(id)
        return id_num

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def prepare_train_data(self, train_path):
        input_names = []
        image1 = []
        image2 = []
        # for dirname in train_path:
        #     print(dirname)
        train_t_gt = train_path + "transmission_layer/"
        train_r_gt = train_path + "reflection_layer/"
        train_b = train_path + "blended/"
        for root, _, fnames in sorted(os.walk(train_t_gt)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path_input = os.path.join(train_b, fname)
                    path_output1 = os.path.join(train_t_gt, fname)
                    path_output2 = os.path.join(train_r_gt, fname)
                    input_names.append(path_input)
                    image1.append(path_output1)
                    image2.append(path_output2)
        return input_names, image1, image2

    def gkern(self, kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel / kernel.max()
        return kernel

    def syn_data(self, t, r, sigma):
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        att = 1.08 + np.random.random() / 10.0

        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        h, w = r_blur.shape[0:2]
        neww = np.random.randint(0, 560 - w - 10)
        newh = np.random.randint(0, 560 - h - 10)
        alpha1 = self.g_mask[newh:newh + h, neww:neww + w, :]
        alpha2 = 1 - np.random.random() / 5.0;
        r_blur_mask = np.multiply(r_blur, alpha1)
        blend = r_blur_mask + t * alpha2

        t = np.power(t, 1 / 2.2)
        r_blur_mask = np.power(r_blur_mask, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        return t, r_blur_mask, blend

    def next_data(self,q):

        for iter in self.ids * config.epoch:
            magic = np.random.random()
            if magic < 0.7:  # choose from synthetic dataset
                    is_syn = True

                    _id_false = int(iter / self.num_train * len(self.syn_image1_list))

                    syn_image1 = cv2.imread(self.syn_image1_list[_id_false], -1)

                    neww = np.random.randint(256, 480)
                    newh = round((neww / syn_image1.shape[1]) * syn_image1.shape[0])

                    output_image_t = cv2.resize(np.float32(syn_image1), (neww, newh), cv2.INTER_CUBIC) / 255.0

                    output_image_r = cv2.resize(np.float32(cv2.imread(self.syn_image2_list[_id_false], -1)), (neww, newh), cv2.INTER_CUBIC) / 255.0


                    file = os.path.splitext(os.path.basename(self.syn_image1_list[_id_false]))[0]
                    sigma = self.k_sz[np.random.randint(0, len(self.k_sz))]

                    _, output_image_r, input_image = self.syn_data(output_image_t, output_image_r, sigma)
            else:  # choose from real dataste
                    is_syn = False

                    _id = int(iter /self.num_train * len(self.input_real_names))

                    inputimg = cv2.imread(self.input_real_names[_id], -1)
                    file = os.path.splitext(os.path.basename(self.input_real_names[_id]))[0]
                    neww = np.random.randint(256, 480)
                    newh = round((neww / inputimg.shape[1]) * inputimg.shape[0])
                    input_image = cv2.resize(np.float32(inputimg), (neww, newh), cv2.INTER_CUBIC) / 255.0
                    output_image_t = cv2.resize(np.float32(cv2.imread(self.output_real_names1[_id], -1)), (neww, newh),
                                                cv2.INTER_CUBIC) / 255.0
                    output_image_r = output_image_t


            if output_image_r.max() < 0.15 or output_image_t.max() < 0.15:
                    print("Invalid reflection file %s (degenerate channel)" % (file))
                    continue
            if input_image.max() < 0.1:
                    print("Invalid file %s (degenerate images)" % (file))
                    continue

            self.input_images.append(cv2.resize(input_image,(self.pic_size,self.pic_size)))
            self.output_images.append(cv2.resize(output_image_t, (self.pic_size, self.pic_size)))
            if(len(self.input_images) == self.batch_size):
               self.input_images = np.array(self.input_images).reshape([self.batch_size, self.pic_size,self.pic_size,3]).astype(np.float32)
               self.output_images = np.array(self.output_images).reshape([self.batch_size, self.pic_size, self.pic_size, 3]).astype(np.float32)

               q.put([self.input_images , self.output_images])
               self.input_images = []
               self.output_images = []
               print("--")












