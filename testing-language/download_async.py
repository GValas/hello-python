import uuid
import asyncio
from aiohttp import ClientSession
import aiofiles
from helpers import timeit

async def hello(url):
    async with ClientSession() as session:
        async with session.get(url) as response:
            print('get...')
            res = await response.read()
            async with aiofiles.open('/tmp/data/' + uuid.uuid4().hex + '.jpg', mode = 'wb+') as f:
                print('write...')
                await f.write(res)

@timeit
def main():
    url = 'http://lorempixel.com/400/200/sports/'
    loop = asyncio.get_event_loop()
    tasks = [hello(url) for i in range(100)]
    loop.run_until_complete(asyncio.wait(tasks))

main()




