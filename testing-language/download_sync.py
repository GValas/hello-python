import uuid
import requests
from helpers import timeit

def hello(url:str)->None:
    print('get...')
    res = requests.get(url)
    with open('/tmp/data/' + uuid.uuid4().hex + '.jpg', mode = 'wb+') as f:
        print('write...')
        f.write(res.content)

@timeit
def main():
    url = 'http://lorempixel.com/400/200/sports/'
    for _ in range(100):
        hello(url)

main()