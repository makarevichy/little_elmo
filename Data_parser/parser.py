import vk
import time
import argparse
import pandas as pd


session = vk.AuthSession('6653822', 'vitaliy.igorevich@mail.ru', 'VitaliyHell1997')
vkapi = vk.API(session)
VERSION = 5.92
COUNT = 50


def drop_except(func):
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except:
                pass
    return wrapper


@drop_except
def get_comments(group_id, post_id, **kwargs):
    return vkapi('wall.getComments',
                 owner_id=group_id,
                 post_id=post_id,
                 v=VERSION,
                 **kwargs)


@drop_except
def get_posts(group_id, **kwargs):
    return vkapi('wall.get',
                 owner_id=group_id,
                 v=VERSION,
                 **kwargs)


def offset(count):
    i = -1
    for i in range(count//COUNT):
        yield i*COUNT, COUNT
    yield (i+1)*COUNT, count%COUNT

def get_all_comments(group_id, post_id):
    count = get_comments(group_id, post_id)['count']
    for start, count_comments in offset(count):
        comments = get_comments(group_id, post_id, 
                                offset=start, count=count_comments)
        for comment in comments['items']:
            yield comment.get('text', '')


def get_posts_id(group_id, count):
    if count == 'all':
         count = get_posts(group_id)['count']
         print(count)
    else:
        count = int(count)
    for start, count_post in offset(count):
        posts = get_posts(group_id, offset=start, count=count_post)
        for post in posts['items']:
            yield post['id']


def join_comments(group_id, count):
    for i, post_id in enumerate(get_posts_id(group_id, count)):
        print(i)
        for text in get_all_comments(group_id, post_id):
            if text:
                yield text


def write(filename, iterator):
    df = pd.DataFrame(enumerate(iterator), columns=['id', 'text'])
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_id')
    parser.add_argument('--count_posts')
    parser.add_argument("--file")
    args = parser.parse_args()
    iterator = join_comments(int(args.group_id), args.count_posts)
    write(args.file, iterator)
    
