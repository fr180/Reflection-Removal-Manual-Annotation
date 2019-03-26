
from multiprocessing import Process,Queue
from dataset_multi_processing import DatasetProcessing as DP
from model import *

if __name__ == '__main__':
    gan = Pix2Pix(batch_size = config.batch_size)
    dp = DP(batch_size = config.batch_size, pic_size=config.image_size)
    q = Queue(12)
    pw = Process(target=dp.next_data,args=(q,))
    pr= Process(target=gan.train,args=(q,))

    pw.start()
    pr.start()
    pw.join()
    pr.terminate()