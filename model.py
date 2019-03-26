import os
from cfg import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
import tensorflow as tf
from tensorflow.python.keras.engine import network
from tensorflow.python.keras.layers import *

import datetime
import time
import numpy as np

# keras.engine.topology.Container
from util import input_fn,calc_ssim,calc_psnr
import cv2

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
train_log_dir = 'weights/logs/train/' + TIMESTAMP
test_log_dir = 'weights/logs/test/'   + TIMESTAMP
class Pix2Pix():
    def __init__(self,batch_size):
        # Input shape
        self.img_rows = config.image_size
        self.img_cols = config.image_size
        self.channels = 3
        self.batch_size = batch_size
        self.img_shape =  (self.img_rows, self.img_cols, self.channels)
        self.mask_shape = (self.img_rows, self.img_cols, 1)


        self.dis_weight = config.dis_weight
        self.gen_weight = config.gen_weight

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        dis_optimizer = tf.train.AdamOptimizer(config.d_lr,0.5)
        optimizer = tf.train.AdamOptimizer(config.g_lr, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        if os.path.exists(self.dis_weight):
            self.discriminator.load_weights(self.dis_weight)

        self.discriminator.compile(loss='mse',
            optimizer=dis_optimizer,
            metrics=['accuracy'])


        self.discriminator_fixed = self.build_discriminator(isnetwork=True)
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        if os.path.exists(self.gen_weight):
            self.generator.load_weights(self.gen_weight)

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        img_mask = Input(shape = self.mask_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator([img_B,img_mask])

        # For the combined model we will only train the generator
        self.discriminator_fixed.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator_fixed([fake_A, img_B, img_mask])

        self.combined = tf.keras.Model(inputs=[img_A, img_B, img_mask], outputs = [valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

        # self.tensorboard.set_model(self.combined)
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.5):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input

        mask = Input(shape=self.mask_shape)
        d0 = Input(shape=self.img_shape)
        combine = Concatenate(axis=-1)([mask, d0])

        # Downsampling
        d1 = conv2d(combine, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)

        gen = tf.keras.Model([d0,mask], output_img)


        return gen


    def build_discriminator(self,isnetwork=False):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        mask = Input(shape=self.mask_shape)

        # Concatenate images and conditioning images by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B,mask])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same',activation="sigmoid")(d4)
        if not isnetwork:
         dis = tf.keras.Model([img_A, img_B,mask], validity)
        else:
         dis = network.Network([img_A,img_B,mask],validity)


        return dis


    def train(self, q):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake =  np.zeros((self.batch_size,) + self.disc_patch)



        val_dataset, data_nums = input_fn(val_batch_size = 20)


        data_nums = config.image_num

        val_iteration = val_dataset.make_initializable_iterator()
        val_input = val_iteration.get_next()

        sess = tf.Session()
        sess.run(val_iteration.initializer)
        val_input = sess.run(val_input)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
        sess.close()


        for epoch in range(config.epoch):
            for batch_i  in range(int(data_nums/self.batch_size)):

                if q.qsize() >= 2:
                    train_batch = q.get(True)
                    imgs_B, imgs_A = train_batch[0], train_batch[1]

                    mask_image = self.get_mask(self.normalizing(imgs_B),self.normalizing(imgs_A))
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Condition on B and generate a translated version

                    fake_A = self.generator.predict([imgs_B, mask_image])

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B, mask_image], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B, mask_image], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    for i in range(5):
                        g_loss = self.combined.train_on_batch([imgs_A, imgs_B,mask_image], [valid, imgs_A])
                        print("g_loss: ",g_loss[0])
                    if batch_i % 8 == 0:
                        average_psnr, average_ssim = self.calc_index(self.normalizing(fake_A), self.normalizing(imgs_A))
                        var_psnr, var_ssim = self.sample_images(epoch, batch_i, val_input,"val")
                        s1 = tf.Summary(value=[tf.Summary.Value(tag="d_loss", simple_value = d_loss[0])])
                        s2 = tf.Summary(value=[tf.Summary.Value(tag="g_loss", simple_value = g_loss[0])])

                        s3 = tf.Summary(value=[tf.Summary.Value(tag="batch_average_psnr", simple_value=average_psnr)])
                        s4 = tf.Summary(value=[tf.Summary.Value(tag="batch_average_ssim", simple_value=average_ssim)])
                        val1 = tf.Summary(value=[tf.Summary.Value(tag="batch_average_psnr", simple_value=var_psnr)])
                        val2 = tf.Summary(value=[tf.Summary.Value(tag="batch_average_ssim", simple_value=var_ssim)])

                        train_writer.add_summary(s1, batch_i + epoch * int(data_nums / self.batch_size))
                        train_writer.add_summary(s2, batch_i + epoch * int(data_nums / self.batch_size))
                        train_writer.add_summary(s3, batch_i + epoch * int(data_nums / self.batch_size))
                        train_writer.add_summary(s4, batch_i + epoch * int(data_nums / self.batch_size))
                        test_writer.add_summary(val1, batch_i + epoch * int(data_nums / self.batch_size))
                        test_writer.add_summary(val2, batch_i + epoch * int(data_nums / self.batch_size))

                        print("batch_i:%d,epoch:%d,batch_psnr:%f,batch_ssim:%f" % (
                            batch_i + 1, epoch, average_psnr, average_ssim))
                    elapsed_time = datetime.datetime.now() - start_time
                    # Plot the progress
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, 100,
                                                                                                          batch_i + 1,
                                                                                                          int(data_nums/self.batch_size),
                                                                                                          d_loss[0],
                                                                                                          100 * d_loss[1],
                                                                                                          g_loss[0],
                                                                                                          elapsed_time))

                    if batch_i % 80 == 0:
                        os.makedirs("weights", exist_ok=True)
                        self.sample_images(epoch, batch_i, train_batch, "train")
                        self.discriminator.save_weights(self.dis_weight,save_format="h5")
                        self.generator.save_weights(self.gen_weight, save_format="h5")
                else:
                    print("wait a moment")
                    time.sleep(10)

    def normalizing(self,input1):
        if np.min(input1) < 0:
           return  (input1 + 1) / 2
        else:
           return input1

    def get_mask(self,input1,input2):
        input1 = input1 * 255.0
        input2 = input2 * 255.0
        imgs_mask = np.zeros([input1.shape[0], self.img_rows, self.img_cols], np.float32)
        mask_image = np.zeros(imgs_mask.shape, np.float32)
        result_image = np.zeros(imgs_mask.shape,np.float32)
        # ---------------------
        for i in range(input1.shape[0]):
            imgs_mask[i] = np.abs(
                np.array(cv2.cvtColor(input2[i].astype(np.uint8), cv2.COLOR_BGR2GRAY),np.float32) -
                np.array(cv2.cvtColor(input1[i].astype(np.uint8), cv2.COLOR_BGR2GRAY),np.float32))
        mask_image[np.where(imgs_mask >= 40)] = 0.8
        mask_image = (mask_image + 0.1)*255.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for index in range(input1.shape[0]):
            term1 = cv2.erode(mask_image[index],kernel,iterations= 1)
            term2 = cv2.dilate(term1,kernel1,iterations = 2)
            result_image[index] = term2

        result_image = np.expand_dims(result_image, axis=-1)
        result_image = result_image
     #   cv2.imwrite("result1.png",result_image[0])
        return result_image

    def get_index(self,chart, nums, threshold):
            area = 0
            for i in reversed(range(255)):
                area = area + chart[i]
                my_threshold = area / nums
                if my_threshold > threshold:
                  #  print(i)
                    return i

    def get_latest_mask(self,input,threshold = 0.3):
        input = input * 255.0


        imgs_mask = np.zeros([input.shape[0], self.img_rows, self.img_cols], np.float32)
        mask_image = np.zeros([input.shape[0], self.img_rows, self.img_cols], np.float32)

        for index in range(input.shape[0]):

            input_gray = (cv2.cvtColor(input[index],cv2.COLOR_BGR2GRAY)).astype(np.uint8)
            input_gray = input_gray.reshape((self.img_rows*self.img_cols))

            light_static_term = np.zeros(256)
            light_static = np.zeros([256, 256])
            light_all = 0
            for i in range(self.img_rows*self.img_cols):

                light_static_term[input_gray[i]] = light_static_term[input_gray[i]] + 1
                light_all = light_all + 1
            input_gray = input_gray.reshape((self.img_rows,self.img_cols))
            imgs_mask[index] = input_gray

            light_static[np.where(input_gray > self.get_index(light_static_term, light_all, threshold))] = 1.0
            mask_image[index] = light_static

        mask_image = np.expand_dims(mask_image, axis=-1)
        #cv2.imwrite("mask.png",(mask_image[0]*255).astype(np.uint8))

        return mask_image


    def sample_images(self, epoch, batch_i, dataset,is_train):
        os.makedirs('images/%s' % (is_train), exist_ok=True)

        imgs_A, imgs_B = dataset[1], dataset[0]

        mask_image = self.get_mask(imgs_B,imgs_A)
        # mask_image = self.get_latest_mask(imgs_B)

        fake_A = self.generator.predict([imgs_B,mask_image])
        fake_A = self.normalizing(fake_A)
        imgs_A = self.normalizing(imgs_A)
        imgs_B = self.normalizing(imgs_B)
        #fake_A = np.minimum(np.maximum(fake_A, 0), 1.0)

        i = 0
        average_psnr, average_ssim = self.calc_index((fake_A*255.0).astype(np.uint8), (imgs_A*255.0).astype(np.uint8))
        if is_train == "val":
            print("val_average_psnr: ", average_psnr, "val_average_ssim: ", average_ssim)
        else:
            print("train_average_psnr: ", average_psnr, "train_average_ssim: ", average_ssim)

        if batch_i % 40 == 0:
            gen_images = np.concatenate([imgs_B, fake_A, imgs_A], axis=2)

            for index in range(gen_images.shape[0]):
                name = "images/%s/%d_%d_%d.png" % (is_train,epoch, batch_i, i)
                name2 = "images/%s/%d_%d_%d_mask.png" % ( is_train, epoch, batch_i, i)
                cv2.imwrite(name, gen_images[index] * 255.0)
                cv2.imwrite(name2,mask_image[index] * 255.0)
                i = i + 1
        return average_psnr, average_ssim

    def predict_images(self,epoch,input_pic,out_dir,load=False):
        os.makedirs('images/%s' %out_dir, exist_ok=True)
        if load:
            self.generator.load_weights(self.gen_weight)
        imgs_A, imgs_B = input_pic[1], input_pic[0]

        mask_image = self.get_mask(imgs_B,imgs_A)
       # latest_mask_image = self.get_latest_mask(imgs_B)

        fake_A = self.generator.predict([imgs_B,mask_image])
        fake_A = self.normalizing(fake_A)
        imgs_A = self.normalizing(imgs_A)
        imgs_B = self.normalizing(imgs_B)

        i = 0
        all_psnr = 0
        all_ssim = 0

        gen_images = np.concatenate([imgs_B, fake_A, imgs_A], axis=2)
        gen_images = gen_images * 255.0


        imgs_A = (imgs_A * 255.0).astype(np.uint8)
        fake_A = (fake_A * 255.0).astype(np.uint8)

        for index in range(gen_images.shape[0]):

            psnr = calc_psnr(imgs_A[index], fake_A[index])
            ssim = calc_ssim(imgs_A[index], fake_A[index])

            all_psnr +=psnr
            all_ssim += ssim
            cv2.imwrite("images/%s/%d_%d.png" % ("predict",epoch,i), gen_images[index])
            print("%d_%d"%(epoch,i),psnr,"--",ssim)
            cv2.imwrite("images/%s/%d_%d_mask.png" % ("predict", epoch, i),mask_image[index]*255.0)
          #  cv2.imwrite("images/%s/%d_%d_real_mask.png" % ("predict", epoch, i), mask_image[index] * 255.0)
            i = i + 1
        return all_psnr,all_ssim

    def calc_index(self,generate,original):

        nums = generate.shape[0]
        all_psnr,all_ssim,i = 0, 0, 0

        for index in range(nums):
            psnr = calc_psnr(generate[index],original[index])
            ssim = calc_ssim(generate[index],original[index])
            all_psnr += psnr
            all_ssim +=ssim
            i = i + 1
        return all_psnr/i,all_ssim/i

    def predict_single_image(self,pic,gt,mask):
        os.makedirs('images/%s' %"predict", exist_ok=True)

        self.generator.load_weights(self.gen_weight)

        pic = (pic/255.0).astype(np.float32)
        gt = (gt / 255.0).astype(np.float32)
        mask = (mask/255.0).astype(np.float32)
        val_mask = self.get_mask(pic,gt)



        fake_A = self.generator.predict([pic,mask])
        fake_val_A=self.generator.predict([pic,val_mask])

        fake_A = self.normalizing(fake_A)
        fake_val_A = self.normalizing(fake_val_A)

      #  fake_A = np.minimum(np.maximum(fake_A, 0), 1.0)
      #  fake_val_A = np.minimum(np.maximum(fake_val_A, 0), 1.0)



        gen_images = np.concatenate([pic, fake_A,gt], axis=2)
        gen_val_images = np.concatenate([pic, fake_val_A,gt], axis=2)



        gen_images = (gen_images * 255.0).astype(np.uint8)
        gen_val_images = (gen_val_images *255.0).astype(np.uint8)

        fake_A = (fake_A * 255.0).astype(np.uint8)
        fake_val_A = (fake_val_A*255.0).astype(np.uint8)

        gt = (gt * 255.0).astype(np.uint8)
        mask = (mask*255.0).astype(np.uint8)
        val_mask= (val_mask*255.0).astype(np.uint8)




        psnr = calc_psnr(gt[0], fake_A[0])
        ssim = calc_ssim(gt[0], fake_A[0])

        psnr_val = calc_psnr(gt[0],fake_val_A[0])
        psnr_ssim = calc_ssim(gt[0],fake_val_A[0])


        cv2.imwrite("images/predict/"+str(0)+"_all.png",gen_images[0])
        cv2.imwrite("images/predict/"+str(0)+"_mask.png",mask[0])
        cv2.imwrite("images/predict/"+str(0)+"_val_all.png",gen_val_images[0])
        cv2.imwrite("images/predict/"+str(0)+"_val_mask.png",val_mask[0])

        return psnr,ssim, psnr_val,psnr_ssim