
from model import *
import math
from util import input_fn
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gan = Pix2Pix(batch_size = config.batch_size)
sess  = tf.Session()
val_dataset, data_nums = input_fn(val_batch_size=20,synthetic=True)

val_iteration = val_dataset.make_initializable_iterator()
sess.run(val_iteration.initializer)
all_psnr,all_ssim = 0, 0
for epoch in range(math.ceil(data_nums/20)):
    val_input = val_iteration.get_next()

    val_input = sess.run(val_input)

    p,s = gan.predict_images(epoch,val_input,"predict",True)
    print("p: ",p,"s: ",s)
    all_psnr += p
    all_ssim += s
    print(epoch)

print("psnr:",all_psnr/(data_nums))
print("ssim:",all_ssim/(data_nums))





####real  psnr: 24.919
        # ssim: 0.8

### fake  psnr: 23.38
        # ssim: 0.853