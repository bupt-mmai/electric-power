import json

import requests
from bs4 import BeautifulSoup


def download(url):
    if url is None:
        return
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36',
        'Accept': 'text / html, application / xhtml + xml, application / xml;q = 0.9, image / webp, image / apng, * / *;q = 0.8, application / signed - exchange;v = b3;q = 0.9',
        'Host': 'baike.baidu.com',
        'Connection': 'keep-alive',
        'Pragma': 'no-cache',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    response.encoding = 'utf-8'
    return response.text


def parse(word, html_content):
    if html_content is None:
        return None
    # 获取html解析器
    soup = BeautifulSoup(html_content, 'lxml')
    if soup.title.string.find('_百度百科')<0 and soup.title.string.find(word) <0:
        return None
    res_dict = {}
    res_dict['title'] = soup.select(".lemmaWgt-lemmaTitle-title > h1")[0].text
    desc_soup = soup.select(
        "body > div.body-wrapper > div.content-wrapper > div > div.main-content > div.lemma-summary > div")
    if desc_soup and len(desc_soup) >= 0:
        res_dict['desc'] = desc_soup[0].text
    item_names = soup.find_all('dt', class_="basicInfo-item name")
    item_values = soup.find_all('dd', class_="basicInfo-item value")  # 找到所有dd标签，返回一个列表
    if item_names and len(item_names) >= 0:
        properties = {}
        for i in range(len(item_names)):
            properties[item_names[i].text] = item_values[i].text.strip()
        res_dict['properties'] = properties
    imgsoup = soup.select(
        'body > div.body-wrapper > div.content-wrapper > div > div.side-content > div.summary-pic > a > img')
    if imgsoup and len(imgsoup) >= 0:
        res_dict['img'] = imgsoup[0].attrs['src']
    paras = soup.find_all('div', class_="para")
    if paras and len(paras) >= 0:
        text_data = ''
        for para in paras:
            text_data += para.text + '\n'
        res_dict['text'] = text_data
    return res_dict


def spider_word(word):
    content = download(f'https://baike.baidu.com/item/{word}')
    res_dict = parse(word, content)
    return res_dict

def write_csv(output_file,line):
    with open(output_file,'a') as fw:
        fw.write(line)

if __name__ == '__main__':
    path = '../entities/entity_x.txt'
    output_file = '../entity_baike/entity_baike_x.csv'
    with open(path, 'r') as fr:
        for word in fr.readlines():
            word = word.strip()
            if word:
                res_dict = spider_word(word)
                print(f'word:{word}' + '\n' + json.dumps(res_dict, indent=4, ensure_ascii=False))
                res_json = json.dumps(res_dict, indent=4, ensure_ascii=False)
                if res_json!='null':
                    write_csv(output_file,line=word+','+json.dumps(res_json,ensure_ascii=False)+'\n')
