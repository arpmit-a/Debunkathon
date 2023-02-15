from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import urllib
import os
Path="/content/Debunkathon/Chromedriver/chromedriver.exe"
os.makedirs("/content/images/")
driver=webdriver.Chrome(executable_path=Path)
def get_img_urls(url):
  image_urls=[]
  driver.get(url)
  i=1
  while i<=5:
    l=driver.find_element(by=By.XPATH,value=f'/html/body/div[7]/div/div[10]/div/div[2]/div[2]/div/div/div/div/div[{i}]/div/div/a/div/div[1]/div/div/img')
    try:
        image_urls.append(l.get_attribute('src'))
        i+=1
    except:
        i+=1
  return image_urls

def download_by_urls(url):
    image_urls=get_img_urls(url)
    for i in range(len(image_urls)):
        urllib.request.urlretrieve(str(image_urls[i]),f"/content/images/image{i}.jpg".format(i))


download_by_urls("https://www.google.com/search?q=The+Battle+of+New+York+Why+This+Primary+Matters&source=lnms&tbm=nws&sa=X&ved=2ahUKEwjZmpa_pZD9AhXmTGwGHe89CWcQ_AUoAXoECAEQAw&biw=858&bih=932&dpr=1")
