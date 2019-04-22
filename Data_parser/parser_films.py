import requests
from bs4 import BeautifulSoup


URL = 'http://subs.com.ru/'


def get_link_pages(count):
    for i in range(252, count+1):
        yield URL+f'list.php?c=rus&d={i}'


def get_page(link):
    return requests.get(link).text


def parse_html(html):
    return BeautifulSoup(html, 'html.parser')


def get_link_film_pages(page_html):
    soup = parse_html(page_html)
    fields = soup.find_all('td', align='center')
    for field in fields:
        link = field.find('a')['href']
        if link.startswith('page'):
            yield URL+link


def get_link_save(page_html):
    soup = parse_html(page_html)
    fields = soup.find_all('td')
    flg = False
    for field in fields:
        if flg:
            return URL + 'sub/rus/' + field.text, field.text
        if field.text.startswith('Имя файла'):
            flg = True


def save(link, filename):
    r = requests.get(link)
    with open(filename, 'wb') as f:
        for chunk in r:
            f.write(chunk)


def main(count):
    for page_link in get_link_pages(count):
        page_html = get_page(page_link)
        for i, link_film_page in enumerate(get_link_film_pages(page_html)):            
            page_film_html = get_page(link_film_page)
            save_link, filename = get_link_save(page_film_html)
            save(save_link, 'rars/'+filename)
            print(i)


if __name__ == '__main__':
    main(500)
