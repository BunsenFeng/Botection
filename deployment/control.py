#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~import & global section~~~~~~~~~~~~~~~~~~~~~~
import torch
import pandas as pd
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pynlpir
import random
import torch.nn.utils.rnn as rnn
import re
import emoji
from selenium import webdriver
from time import sleep
import random
import urllib.request
import json
import re
import requests
import time
import xlrd
import re
import openpyxl
from selenium import webdriver
from time import sleep
import re
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import pickle
na = 'a'
# 设置代理IP
iplist = ['112.228.161.57:8118', '125.126.164.21:34592', '122.72.18.35:80', '163.125.151.124:9999',
          '114.250.25.19:80']
proxy_addr = "125.126.164.21:34592"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~import & globa end~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~idextract begin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def idextract(name):
    option = webdriver.ChromeOptions()
    option.add_argument('disable-infobars')
    driver = webdriver.Chrome(r'chromedriver.exe')
    driver.get('https://s.weibo.com/user?q='+name)
    sleep(2)
    links = driver.find_element_by_xpath('//*[@id="pl_user_feedList"]/div/div[2]/div/a[1]')
    links.click()
    sleep(8)
    handles = driver.window_handles
    driver.switch_to.window(handles[-1])
    sleep(2)
    text = driver.page_source
    driver.quit()
    st = text.find('$CONFIG[\'oid\']=\'')
    uid = ''
    for i in range(10):
        uid += text[st+16+i]
    return uid
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~idextract end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~crawler begin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def use_proxy(url,proxy_addr):

    time.sleep(0.5)
    req=urllib.request.Request(url)
    req.add_header("User-Agent","Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0")
    proxy=urllib.request.ProxyHandler({'http':random.choice(iplist)})
    opener=urllib.request.build_opener(proxy,urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    data=urllib.request.urlopen(req).read().decode('utf-8','ignore')
    return data
def get_containerid(url):
    na = 'a'
    # 设置代理IP

    iplist = ['112.228.161.57:8118', '125.126.164.21:34592', '122.72.18.35:80', '163.125.151.124:9999',
              '114.250.25.19:80']
    data=use_proxy(url,random.choice(iplist))
    content=json.loads(data).get('data')
    for data in content.get('tabsInfo').get('tabs'):
        if(data.get('tab_type')=='weibo'):
            containerid=data.get('containerid')
    return containerid
def get_userInfo(id,file):
    na = 'a'
    # 设置代理IP

    iplist = ['112.228.161.57:8118', '125.126.164.21:34592', '122.72.18.35:80', '163.125.151.124:9999',
              '114.250.25.19:80']

    proxy_addr = "125.126.164.21:34592"
    url='https://m.weibo.cn/api/container/getIndex?type=uid&value='+id
    data=use_proxy(url,random.choice(iplist))
    ifok=int(json.loads(data).get('ok'))
    if ifok==0:
        return 0
    f=open(file, "w")
    f.truncate()
    print(id+"start")
    content=json.loads(data).get('data')
    profile_image_url=content.get('userInfo').get('profile_image_url')
    description=content.get('userInfo').get('description')
    profile_url=content.get('userInfo').get('profile_url')
    verified=content.get('userInfo').get('verified')
    guanzhu=content.get('userInfo').get('follow_count')
    name=content.get('userInfo').get('screen_name')
    na=name
    fensi=content.get('userInfo').get('followers_count')
    gender=content.get('userInfo').get('gender')
    urank=content.get('userInfo').get('urank')
    with open(file,'a',encoding='utf-8') as fh:
        fh.write("微博昵称："+name+"\n"+"微博主页地址："+profile_url+"\n"+"微博头像地址："+profile_image_url+"\n"+"是否认证："+str(verified)+"\n"+"微博说明："+description+"\n"+"关注人数："+str(guanzhu)+"\n"+"粉丝数："+str(fensi)+"\n"+"性别："+gender+"\n"+"微博等级："+str(urank)+"\n")
    return 1
def get_weibo(id, file):
    i = 1
    Directory = 'D:\weibo\weibo'
    while True:
        url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
        weibo_url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id + '&containerid=' + get_containerid(
            url) + '&page=' + str(i)
        try:
            data = use_proxy(weibo_url, random.choice(iplist))
            content = json.loads(data).get('data')
            cards = content.get('cards')
            if (len(cards) > 0):
                print("---正在爬取第" + str(i) + "页")
                for j in range(len(cards)):
                    # print("-----正在爬取第"+str(i)+"页，第"+str(j)+"条微博------")
                    card_type = cards[j].get('card_type')
                    if (card_type == 9):
                        mblog = cards[j].get('mblog')
                        #  #print(mblog)
                        #   #print(str(mblog).find("转发微博"))
                        n = 0
                        if str(mblog).find('retweeted_status') == -1:
                            if str(mblog).find('original_pic') != -1:
                                img_url = re.findall(r"'url': '(.+?)'", str(mblog))  ##pics(.+?)
                                #          timename = str(time.time())
                                #          timename = timename.replace('.', '')
                                #         timename = timename[7:]#利用时间作为独特的名称
                                for url in img_url:
                                    #           print('第' + str(n) + ' 张', end='')
                                    #          with open(Directory + timename+url[-5:], 'wb') as f:
                                    #             f.write(requests.get(url).content)
                                    #        print('...OK!')
                                    n = n + 1
                            # # if( n%3==0 ):  ##延迟爬取，防止截流
                            #   #  time.sleep(3)

                        pic_count = n
                        attitudes_count = mblog.get('attitudes_count')
                        comments_count = mblog.get('comments_count')
                        created_at = mblog.get('created_at')
                        reposts_count = mblog.get('reposts_count')
                        scheme = cards[j].get('scheme')
                        text = mblog.get('text')
                        with open(file, 'a', encoding='utf-8') as fh:
                            fh.write("----第" + str(i) + "页，第" + str(j) + "条微博----" + "\n")
                            fh.write("微博地址：" + str(scheme) + "\n" + "发布时间：" + str(
                                created_at) + "\n" + "微博内容：" + text + "\n" + "点赞数：" + str(
                                attitudes_count) + "\n" + "评论数：" + str(comments_count) + "\n" + "转发数：" + str(
                                reposts_count) + "\n" + "图片数：" + str(pic_count) + "\n")
                i += 1
            else:
                break
        except Exception as e:
            print(e)
            pass
def crawler(ind):
    file='data\weibo'+ind+".txt"
    continum=get_userInfo(ind,file)
    if continum==1:
        get_weibo(ind,file)
    print(str(ind)+" finish")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~crawler end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~wash begin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def wash(userid):
    data_final = ''
    position = '.\\data\\weibo' + userid + '.txt'
    with open(position, "r", encoding='utf-8') as f:  # 打开文件
        data_raw = f.read()  # 读取文件
        wnn = 1
        while wnn:
            head = data_raw.find('微博内容：') + 5
            tail = data_raw.find('点赞数：')
            data = data_raw[head: tail]
            data_raw = data_raw[tail + 4:]
            if data_raw.find('微博内容：') == -1:  # 后面没有新微博的话停止这个用户的数据读取
                wnn = 0
            if (data[0: 2] == '//') or \
               (data == '转发微博\n') or \
               (data == 'Repost\n'):  # 纯转发的删删掉
                continue
            if data.find(r"//<a href='/n/") != -1:  # 转发别人的内容删删掉
                data = data[0: data.find(r"//<a href='/n/")]
            data = data.replace(r"<br />", " ")  # 先处理转行

            # 处理所有a标签的（我真是仏了）
            head = data.find(r"<a ")
            while head != -1:
                tail = data.find(r"</a>") + 4
                sub_data = data[head: tail]
                if sub_data[0: 16] == r"<a href=" + r"/status/":  # 先删掉展开全文
                    data = data.replace(sub_data, '')
                    head = data.find(r"<a ")
                    continue
                if sub_data[0: 12] == r"<a href='/n/":  # 处理at
                    data = data.replace(sub_data, ' ttttt ')
                    head = data.find(r"<a ")
                    continue
                if sub_data.find('>#') != -1:  # 处理hashtags
                    tep_data = sub_data[sub_data.find('>#') + 2:]
                    sub_text = tep_data[0: tep_data.find('#')]
                    data = data.replace(sub_data, ' ' + sub_text + ' ggggg ')
                    head = data.find(r"<a ")
                    continue
                data = data.replace(sub_data, ' uuuuu ')
                head = data.find(r"<a ")
                continue

            # 处理表情
            head = data.find(r"<span ")
            while head != -1:
                tail = data.find(r"</span>") + 7
                sub_data = data[head: tail]
                data = data.replace(sub_data, ' eeeee ')
                head = data.find(r"<span ")

            # 处理emoji
            data = emoji.demojize(data)
            data = re.sub(r':(.*_.*):', ' eeeee ', data)

            # 结束啦~
            data_final = data_final + data

    # 写回文件
    new_pos = '.\\data\\' + userid + '.txt'
    with open(new_pos, "w", encoding='utf-8') as f:
        if data_final != '':  # 保证不会写回空时出错（好像本身也不会出错的样子）
            f.write(data_final)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~wash end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~textual begin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, p, n):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, dropout=p, num_layers = n)
        self.hidden2label = nn.Linear(2*hidden_dim, label_size)
        self.num_layer = n
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2*self.num_layer, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(2*self.num_layer, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        #self.lstm1.flatten_parameters()
        #self.lstm2.flatten_parameters()
        #self.lstm3.flatten_parameters()
        embeds = self.word_embeddings(sentence)
        #print(embeds.shape)
        a = embeds.clone().view(len(sentence), self.batch_size, -1)
        #print(a.shape)
        lstm_out, self.hidden = self.lstm(a, self.hidden)
        temp = (lstm_out[-1]).clone()
        y = self.hidden2label(temp)
        log_probs = F.log_softmax(y, dim = 1)
        #log_probs = F.softmax(y, dim = 1)
        return log_probs
def more_than(arr, num):
    temp = 0
    for x in arr:
        temp += (x > num)
    return temp
def less_than(arr, num):
    temp = 0
    for x in arr:
        temp += (x < num)
    return temp
def textual(uid):
    import torch
    import pynlpir
    import random
    import numpy as np
    torch.set_num_threads(8)
    torch.manual_seed(1)
    random.seed(1)
    # opening embedding file
    print('opening embedding file')
    f = open('sgns_weibo.bigram-char', 'r', encoding='utf8')
    raw = f.readlines()
    f.close()

    # constructing word to index dictionary
    print('constructing word to index dictionary')
    word_to_ix = dict()
    iter = 0
    for line in raw:
        word_to_ix[(line.split())[0]] = iter
        iter = iter + 1

    for i in ['ttttt', 'ggggg', 'uuuuu', 'eeeee', 'ooooo', ' ']:
        word_to_ix[i] = iter
        iter += 1

    model_path = 'mr_best_model_minibatch_acc_7863.model'
    EMBEDDING_DIM = 300
    VOCAB_SIZE = 195203
    HIDDEN_DIM = 100  # TO BE TUNED
    LABEL_SIZE = 2
    BATCH_SIZE = 1  # TO BE TUNED
    EPOCH = 100  # TO BE TUNED
    DROPOUT = 0.5  # TO BE TUNED 0.95* every epoch
    NUM_LAYER = 2  # TO BE TUNED
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, LABEL_SIZE, BATCH_SIZE, DROPOUT, NUM_LAYER)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    path = './data/' + uid + '.txt'

    text_set = []
    pynlpir.open()
    f = open(path, 'r', encoding='utf8')
    raw = f.readlines()
    f.close()
    if len(raw) == 0:
        return [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] #to be determined

    for line in raw:
        #print(line)
        if (len(line) == 0):
            continue
        # tokenizer from chinese academy of sciences
        temp = pynlpir.segment(line, pos_tagging=False)  # 使用pos_tagging来关闭词性标注
        temp2 = [x for x in temp if x != ' ']  # remove redundant spaces

        sentence = []
        for word in temp2:
            if word in word_to_ix.keys():
                sentence.append(word_to_ix[word])
            else:
                sentence.append(word_to_ix['ooooo'])  # oov

        text_set.append(sentence)
    pynlpir.close()

    scores = []
    model.eval()
    if len(text_set) == 0:
        return [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for sent in text_set:
        if(len(sent) == 0):
            continue
        model.hidden = model.init_hidden() #detach from last example
        #print(sent)
        temp = model(torch.tensor(sent))
        temp = (temp[0][1]).item()
        scores.append(np.e ** temp)

    #print(scores)
    feature = []
    # mean >0.8 >0.9 <0.1 <0.2 max min median  continuous>0.75 continuous<0.25
    feature.append(np.mean(scores))
    feature.append(more_than(scores, 0.8))
    feature.append(more_than(scores, 0.9))
    feature.append(less_than(scores, 0.1))
    feature.append(less_than(scores, 0.2))
    feature.append(max(scores))
    feature.append(min(scores))
    feature.append(np.median(scores))

    count = 0
    temp = 0
    for x in scores:
        if x > 0.75:
            temp += 1
        else:
            count = max(count, temp)
            temp = 0
    count = max(count, temp)
    feature.append(count)

    count = 0
    temp = 0
    for x in scores:
        if x < 0.25:
            temp += 1
        else:
            count = max(count, temp)
            temp = 0
    count = max(count, temp)
    feature.append(count)
    return feature
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~textual end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~meta begin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def if_leap_year(year):
    if (year % 4 == 0) and (year % 100 != 0):
        return 1
    return 0
def cal_day(strs):
    cnt = 0
    mouth_day = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    year = int(strs[0:4])
    mouth = int(strs[4:6])
    day = int(strs[6:8])
    cnt += day
    while mouth != 1:
        cnt += mouth_day[mouth - 1]
        if mouth - 1 == 2:
            cnt += if_leap_year(year - 1)
        mouth -= 1
    while year != 2010:
        cnt += 365 + if_leap_year(year - 1)
        year -= 1
    return cnt
def get_strtime(text):
    regex_list = [
        "(\d{4}-\d{1,2}-\d{1,2})"
    ]
    for reg in regex_list:
        t = re.search(reg,text)
        if t:
            t = t.group(1)
            return t
def get_reg(uid):
    driver = webdriver.Chrome(r'chromedriver.exe')
    driver1 = webdriver.Chrome(r'chromedriver.exe')
    driver.get('https://weibo.com/u/'+uid)
    sleep(10)
    text = driver.page_source
    st = text.find('$CONFIG[\'page_id\']=\'')
    driver.quit()
    pid = ''
    for i in range(16):
        pid += text[st+20+i]
    print(pid)
    sleep(2)
    driver1.get('https://weibo.com/p/'+pid+'/info?mod=pedit_more')
    sleep(10)
    text = driver1.page_source
    driver1.quit()
    return get_strtime(text).replace('-','')
def find_cnt(str1, str2):
    num = (len(str1) - len(str1.replace(str2, ""))) // len(str2)
    return num
def get_other_info(index):
    if len(index) == 9:
        index = '0' + index
    f = open('data/' + index + '.txt', 'r', encoding='UTF-8')
    list = f.readlines()
    tot_us = 0
    tot_es = 0
    tot_gs = 0
    tot_ts = 0
    for strs in list:
        tot_us += find_cnt(strs, 'uuuuu')
        tot_es += find_cnt(strs, 'eeeee')
        tot_gs += find_cnt(strs, 'ggggg')
        tot_ts += find_cnt(strs, 'ttttt')
    tmp = []
    tmp.append(tot_us)
    tmp.append(tot_es)
    tmp.append(tot_gs)
    tmp.append(tot_ts)

    f.close()
    return tmp
def get_info(index):
    if len(index) == 9:
        index = '0' + index
    print(index)
    f = open('data/weibo' + index + '.txt', 'r', encoding='UTF-8')
    list = f.readlines()
    tot_weibo_num = 0
    tot_like_num = 0
    tot_com_num = 0
    tot_ret_num = 0
    tot_pic_num = 0
    active_day = -cal_day(get_reg(index))
    for str in list:
        if str.find('关注人数：') == 0:
            num = re.findall('(\d+)', str)
            following_num = int(num[0])
        if str.find('粉丝数：') == 0:
            num = re.findall('(\d+)', str)
            follower_num = int(num[0])
        if '----' in str:
            tot_weibo_num += 1
        if str.find('点赞数：') == 0:
            num = re.findall('(\d)', str)
            tot_like_num += int(num[0])
        if str.find('评论数：') == 0:
            num = re.findall('(\d)', str)
            tot_com_num += int(num[0])
        if str.find('转发数：') == 0:
            num = re.findall('(\d)', str)
            tot_ret_num += int(num[0])
        if str.find('图片数：') == 0:
            num = re.findall('(\d)', str)
            tot_pic_num += int(num[0]) // 2
        if str.find('发布时间：') == 0:
            if '前' in str or '昨天' in str or '刚刚' in str:
                time = '20200705'
            else:
                time = str.replace('发布时间：', '')
                time = time.replace('-', '')
                if len(time) == 5:
                    time = '2020' + time
            num = cal_day(time)
            if num > active_day:
                active_day = num
    tmp = []
    tmp.append(following_num)
    tmp.append(follower_num)
    tmp.append(tot_weibo_num)
    tmp.append(tot_like_num)
    tmp.append(tot_com_num)
    tmp.append(tot_ret_num)
    tmp.append(tot_pic_num)
    tmp.append(active_day)
    return tmp
def meta(uid):
    return get_info(uid) + get_other_info(uid)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~meta end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~forest begin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_rate(p1,p2,x):
    if p1>p2:
        if (x<=p1) and (x>=p2):
            return 0.6*x/(p1-p2)+0.2-0.6*p2/(p1-p2)
        if (x>p1):
            return 1-0.2*math.exp(p1-x)
        if (x<p2):
            return 0.2*math.exp(x-p2)
    else:
        if (x<=p2) and (x>=p1):
            return 0.6/(p1-p2)*x-0.6/(p1-p2)*p1+0.8
        if (x<p1):
            return 1-0.2*math.exp(x-p1)
        if (x>p2):
            return 0.2*math.exp(-x+p2)
def forest(feature):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    botnum = [59.5,2,46,0,0,0,0,3340.5,1,9.5,0,0]
    notbotnum = [112,30,32.5,3,3,1,7,3431,8,5,3,2]
    meta_ratio = 0
    tmp = []
    for i in range(12):
        tmp.append(get_rate(botnum[i],notbotnum[i],feature[i]))
        meta_ratio += tmp[i]
    meta_ratio /= 12
    text_ratio = feature[12]
    feature = [feature]
    result = model.predict(feature)
    trees = model.estimators_
    botnum = 0
    for index, trs in enumerate(trees):
        if trs.predict(feature) == 1:
             botnum += 1
    return [meta_ratio,text_ratio,result,botnum/25,tmp]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~forest end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def control(username):
    uid = idextract(username)
    crawler(uid)
    wash(uid)
    x = textual(uid)
    x = x[:-1]
    y = meta(uid)
    return forest(y+x) # return result & values for interpretability

print(control('_-张小雪'))