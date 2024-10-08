from SwinDir.main import main as sr
from LLIE.test import main as le
# from yolov8.detect_rec_plate import main as rc
def image_save(image):
    image.save("../temp/original/original.png")
def test():
    print("开始处理")
    le()
    print("增强完成")
    enhanced_img = sr()
    print("finished")
    # recimg, recrlt = rc()
    return enhanced_img

if __name__ == '__main__':
    print("Hello, World!")
    test()