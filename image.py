from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import urllib
import os
Path="/content/Debunkathon/Chromedriver/chromedriver.exe"
os.makedirs("/content/images/")
options = webdriver.ChromeOptions()
#run Selenium in headless mode
options.add_argument('--headless')
options.add_argument('--no-sandbox')
#overcome limited resource problems
options.add_argument('--disable-dev-shm-usage')
options.add_argument("lang=en")
#open Browser in maximized mode
options.add_argument("start-maximized")
#disable infobars
options.add_argument("disable-infobars")
#disable extension
options.add_argument("--disable-extensions")
options.add_argument("--incognito")
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=options)
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
