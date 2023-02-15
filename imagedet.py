import argparse
import image
import cv2
from PIL import Image, ImageFilter
import preprocess
import logging

def save_images(title):
    title = remove_punctuations(title)
    title=title.replace(" " ,"+")
    #we will search google news page with this titles.
    download_by_urls(f"https://www.google.com/search?q={title}&source=lnms&tbm=nws&sa=X&ved=2ahUKEwjZmpa_pZD9AhXmTGwGHe89CWcQ_AUoAXoECAEQAw&biw=858&bih=932&dpr=1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--title", help="Enter your title",default='')
    parser.add_argument("-i", "--img", help="Enter your news image path",default='')
    args = parser.parse_args()
    im = Image.open(args.img)
    save_images(args.title)
    logging.info("The input image is this:")
    im.show()
    logging.info("Images got after reverse image search are:")
    img1=Image.open("/content/images/image1.jpg")
    img2=Image.open("/content/images/image1.jpg")
    img3=Image.open("/content/images/image1.jpg")
    img4=Image.open("/content/images/image1.jpg")
    img5=Image.open("/content/images/image1.jpg")
    img1.show()
    img2.show()
    img3.show()
    img4.show()
    img5.show()

