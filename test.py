import os
import cv2
file_dir = "test/synthetic/data"
file = os.listdir(file_dir)
i = 1
for f in file:
    num = f.split("_")[0]
    input_pic_dir = os.path.join(file_dir, f)
    gt_pic_dir = "test/synthetic/gt/"+str(num)+"_clean.png"

    input_pic = cv2.imread(input_pic_dir)
    gt_pic = cv2.imread(gt_pic_dir)



    cv2.imwrite("test/synthetic/data1/"+"/"+str(i)+".png",input_pic)
    cv2.imwrite("test/synthetic/gt1/"+str(i) + ".png", gt_pic)
    i = i + 1