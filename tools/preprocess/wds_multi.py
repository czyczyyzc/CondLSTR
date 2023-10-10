import math
import os
import os.path as osp
import random
import argparse
import webdataset as wds
from glob import glob
import json
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser(description='Webdataset convertion')
parser.add_argument('-i', '--data-dir', metavar='DIR', help='path to imagenet')
parser.add_argument('-o', '--output-dir', metavar='DIR', help='path to output folder')
parser.add_argument('-j', '--num-processes', type=int, default=32, help='number of processes')
parser.add_argument('-n', '--maxcount', type=int, default=2000, help='number of processes')
parser.add_argument('--seed', type=int, default=100, help='number of seed')


def readfile(fname):
    with open(fname, "rb") as stream:
        return stream.read()


def process_chunk(args, chunk, start_index, chunk_index):
    os.makedirs(args.output_dir, exist_ok=True)
    with wds.ShardWriter(osp.join(args.output_dir, '{:02d}_%05d.tar'.format(chunk_index)), maxcount=args.maxcount) as sink:
        for idx, item in enumerate(chunk):
            try:
                img_name, json_name = item.split(',')
                image = readfile(osp.join(args.data_dir, 'files', img_name))
                with open(osp.join(args.data_dir, 'files', json_name), "r") as f:
                    meta = json.load(f)
                key = os.path.splitext(json_name)[0]
                sample = {"__key__": key, "jpg": image, "json": meta}
                sink.write(sample)
            except:
                print(f"corrupted sample at global index {start_index + idx}, skip.")


def wds_shards_create(args):
    
    with open(osp.join(args.data_dir, "valid_pair.txt"), "r") as f:
        dataset = f.read()
        dataset = dataset.split('\n')[:-1]
    
    print(len(dataset))
    random.seed(args.seed)

    indexes = list(range(len(dataset)))
    random.shuffle(indexes)
    dataset = [dataset[i] for i in indexes]
    
    # Split dataset into chunks for multiprocessing
    chunk_size = math.ceil(len(dataset) / (args.num_processes * args.maxcount)) * args.maxcount
    chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]

    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = [executor.submit(process_chunk, args, chunk, i*chunk_size, i) for i, chunk in enumerate(chunks)]
        for future in futures:
            future.result()
    
    file_list = glob(osp.join(args.output_dir, '*.tar'))
    file_list = sorted(file_list)
    for i, file_path in enumerate(file_list):
        os.rename(file_path, '/'.join(file_path.split('/')[:-1] + ['{:05d}.tar'.format(i)]))
    print('finish dataset')


if __name__ == '__main__':
    args = parser.parse_args()
    wds_shards_create(args)
