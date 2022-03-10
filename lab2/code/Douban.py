from urllib.request import urlopen, Request, urlretrieve
from bs4 import BeautifulSoup
import os
import re
import json
import time
import pdb



"""
爬取豆瓣top250的250部电影，以下是一步步从raw_url(也就是'https://movie.douban.com/')
爬取top250部电影特定内容和封面的代码，附详细注释。
"""



def general_url_process(url, headers):
    """
    通过代理访问url，返回源代码并解码
    返回html格式的对象
    """
    proxy = Request(url = url,headers = headers)            # 设置代理
    raw_content = urlopen(proxy)                            # 返回网址的源代码
    content = raw_content.read().decode('utf-8')            # 以utf-8格式解码
    pagehtml = BeautifulSoup(content, 'lxml')               # 解析源代码
    return pagehtml

def get_toplist_url(url, headers):
    """
    由"douban.com"找到"/chart"
    返回douban/chart排行榜的url
    """
    pagehtml = general_url_process(url = url, headers = headers)
    upline = pagehtml.find('div', class_='nav-items')       # 找到顶栏
    upline_element = upline.find_all('a')                   # 找到顶栏中的所有元素
    url_toplist = upline_element[3].get('href')             # 定位到“排行榜”的url
    return url_toplist                                      # 返回“排行榜”的url

def get_top250_url(url, headers):
    """
    由"/chart"找到"/top250",也就是目标爬取的电影list的网址
    返回top250第一页的url
    """
    pagehtml = general_url_process(url = url, headers = headers)
    top250 = pagehtml.find('div', class_='douban-top250-hd')# 找到"top250"
    url_top250 = top250.find('a').get('href')               # 定位到"top250"的url
    return url_top250                                       # 返回"top250"的url

def get_total_pages_list(url, headers):
    """
    由top250第一页的url得到一共10页的url（每页25部电影）
    返回一个list，含有10个页面的url
    """
    total_page_list = []

    total_page_list.append(url)
    pagehtml = general_url_process(url = url, headers = headers)
    total_page = pagehtml.find('div',class_='paginator')    # 找到所有10个页面的class
    page_id = total_page.find_all('a')                      # 获取所有page_id
    # pdb.set_trace()
    length = len(page_id)
    for i in range(length-1):
        cur_url = page_id[i].get('href')
        new_url = url + cur_url                             # 返回url
        total_page_list.append(new_url)
    return total_page_list

def get_pic_and_info(list):
    """
    获取电影的海报url和movie url
    """
    pic_url = []
    info_url = []
    for i in range(len(list)):
        cur_info = list[i]
        pic = cur_info.find('img').get('src')               # 找到pic_url
        info = cur_info.find('a').get('href')               # 找到info_url
        pic_url.append(pic)
        info_url.append(info)
    return pic_url,info_url

def get_each_page_pic_and_info(url, headers):
    """
    获取一个页面25部电影的海报urls和movie urls
    """
    pic_url = []
    info_url = []
    pagehtml = general_url_process(url = url, headers = headers)
    item_list = pagehtml.find_all('div', class_='pic')       # 获取item的pic和info
    pic_url, info_url = get_pic_and_info(list = item_list)
    return pic_url,info_url

def get_all_pic_and_movie_urls(movie_path, pic_path, url, headers):
    """
    获取top250部电影的海报urls和movie urls，存入目标路径
    """
    all_pic_url = []
    all_movie_url = []

    toplist_url = get_toplist_url(url = raw_url, headers = my_headers)
    top250_url = get_top250_url(url = toplist_url, headers = my_headers)
    total_page_list = get_total_pages_list(url = top250_url, headers = my_headers)

    for i in range(len(total_page_list)):
        print("Save page: " + str(i+1))
        cur_page = total_page_list[i]
        pic, movie_url = get_each_page_pic_and_info(url = cur_page, headers = my_headers)
        all_pic_url.extend(pic)
        all_movie_url.extend(movie_url)

    f1 = open(save_movie_urls_path,"w")
    for line in all_movie_url:
        f1.write(line+'\n')
    f1.close()
    print("Save all movie urls")

    f2 = open(save_pic_urls_path,"w")
    for line in all_pic_url:
        f2.write(line+'\n')
    f2.close()
    print("Save all pics urls")

def read_pic_and_movie_urls(movie_path, pic_path):
    """
    载出已保存好的top250 urls
    """
    all_pic_urls = []
    all_movie_urls = []
    f1 = open(movie_path,"r")
    f2 = open(pic_path,"r")
    all_movie_urls = f1.read().splitlines()
    all_pic_urls = f2.read().splitlines()
    f1.close()
    f2.close()
    return all_pic_urls, all_movie_urls

def get_each_movie_info(url, headers):
    """
    返回：片名，{片名 导演 编剧 主演 类型 官方网站 制片地区 语言 上映日期 片长 评分}
    """

    info_dict = {}

    pagehtml = general_url_process(url = url, headers = headers)
    name = pagehtml.find(attrs={'property': 'v:itemreviewed'}).text         # 获取片名
    print("Start getting info of " + name)
    info_dict['片名'] = name
    score = pagehtml.find(attrs={'property': 'v:average'}).text             # 获取评分
    infos = pagehtml.find(attrs={'id': 'info'}).text.split('\n')[1:-3]      # 获取相关信息text
    for i in range(len(infos)):
        info = infos[i].strip(" ").split(":")
        key = info[0]
        value = info[1].strip(" ").split(" / ")
        if len(value) == 1:
            value = str(value[0])
        info_dict[key] = value
    info_dict['评分'] = float(score)
    return name, info_dict

def save_pic(pic_url, pic_name, save_dir):
    """
    将picture存进save_dir中
    输入：pic_url, pic_name, save_dir
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time.sleep(1)                   # 防止ip被封
    image = pic_url
    save_path = save_dir + '/' + pic_name + '_' + str(i+1) + '.jpg'
    if not os.path.exists(save_path):
        urlretrieve(image, save_path)   # 利用urllib.request.urltrieve方法下载图片

def save_json(path, dict):
    """
    将dict对象存入指定path
    """
    file = open(path, "a", encoding="utf-8")
    buffer = json.dumps(dict, ensure_ascii=False)
    print("Start Saving...")
    file.write(buffer)
    file.write("\n")
    file.close()
    print("End Saving!")

if __name__ == '__main__':
    raw_url = 'https://movie.douban.com/'
    my_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36 Edg/98.0.1108.62'}
    
    save_pic_dir = './pictures'
    save_movie_urls_path = './movie_urls.txt'
    save_pic_urls_path = './pic_urls.txt'
    save_result_file_path = "./result.json"
    

    if not os.path.exists(save_pic_dir):
        os.makedirs(save_pic_dir)

    # 测试时可以删去"movie_url.txt"和"pic_urls.txt"两个文件，来检查用来爬取网址的代码的可行性，
    # 否则会自动选取已经提前爬好并存放在这两个文件中的数据，以避免反复访问造成的可能的ip误封和流量消耗。
    if not (os.path.exists(save_movie_urls_path) and os.path.exists(save_pic_urls_path)):
        print("Haven't downloaded before ")
        print("Start downloading urls...")
        get_all_pic_and_movie_urls(movie_path = save_movie_urls_path,pic_path = save_pic_urls_path,url = raw_url,headers = my_headers)
        print("Download successfully")
    else:
        print("Have downloaded before, stored in "+ save_movie_urls_path +" and " + save_pic_urls_path)
        print("Start reloading urls...")
        all_pic_urls, all_movie_urls = read_pic_and_movie_urls(movie_path = save_movie_urls_path, pic_path = save_pic_urls_path)
        print("Reload successfully")

    # 主循环，对每部电影进行处理
    for i in range(len(all_movie_urls)):
        time.sleep(1)       # 防止ip被封
        print(str(i+1))
        cur_name, cur_dict = get_each_movie_info( url = all_movie_urls[i], headers = my_headers)
        # 已保存，跳过json保存过程。测试时删除"result.json"即可。
        if not os.path.exists(save_result_file_path):
            save_json(path = save_result_file_path, dict = cur_dict)
        save_pic(pic_url = all_pic_urls[i], pic_name = cur_name, save_dir = save_pic_dir)

    print("Finished!")


