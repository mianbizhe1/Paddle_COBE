from SwinDir.main import main as sr
from LLIE.test import main as le
from yolov8.detect_rec_plate import main as rc
def image_save(image):
    image.save("../temp/original/original.png")
def console():
    print("开始处理")
    le()
    print("增强完成")
    enhanced_img = sr()
    print("重建完成")
    recimg, recrlt = rc()
    print("rec success")
    return enhanced_img, recimg, recrlt

if __name__ == '__main__':
    print("Hello, World!")
    console()