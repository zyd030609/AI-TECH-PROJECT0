import pickle
from config import CONFIG
import redis

def get_redis_cli(): #连接数据库并返回实例
    r = redis.StrictRedis(host=CONFIG['redis_host'], port=CONFIG['redis_port'], db=CONFIG['redis_db'])
    return r
def get_list_range(redis_cli,name,l,r=-1): #从数据库读取所需数据，然后反序列化，=>list
    assert isinstance(redis_cli,redis.Redis)
    list = redis_cli.lrange(name,l,r)
    return [pickle.loads(d) for d in list]

if __name__ == '__main__':
    r = get_redis_cli()
    with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
        data_file = pickle.load(data_dict)
        data_buffer = data_file['data_buffer']
    for d in data_buffer:
        r.rpush('train_data_buffer',pickle.dumps(d)) #将缓冲存入数据库
    # r.rpush('test',pickle.dumps(([8,2],[2,4],5)))
    # p = get_list_range(r,'test',0,-1)
    # print(p)
