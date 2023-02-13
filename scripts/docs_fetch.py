#!/usr/bin/env python3

import os
import sys
import stat
import tempfile
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch docs from other repos and build them.')

    parser.add_argument('--config', type=str, default='config.json',
                        help='config file path')

    parser.add_argument('--dir', type=str, default='docs',
                        help='docs directory')

    args = parser.parse_args()

    return args


def main(config_file, docs_dir):

    tmp_path = tempfile.mkdtemp()

    with open(config_file, 'r') as load_f:
        json_dict = json.load(load_f)

        for doc in json_dict['docs']:
            url = doc['url']
            branch = doc['branch']
            catalog = doc['catalog']
            languages = doc['languages']
            name = doc['url'].split('/')[-1].split('.')[0]
            work_path = os.path.join(tmp_path, name)
            os.system("git clone -b {} {} {}".format(branch, url, work_path))

            # copy docs
            for language in languages:

                docs_path = os.path.join(docs_dir, language, catalog)
                index_path = os.path.join(docs_path, 'index.rst')
                
                # create index
                if not os.path.exists(index_path):
                    os.system("mkdir -p {}" .format(docs_path))
                    os.system("touch {}" .format(index_path))
                    with open(index_path, 'w') as f:
                        f.write('************\r{}\r************\r\r'.format(catalog))
                        f.write('.. toctree::\r    :maxdepth: 1\r    \r\r')
                    
                # copy static resources
                os.system(
                    "cp -r {}/docs/_static {}" .format(work_path, docs_path))
                
                # create example dir
                os.system(
                    "mkdir -p {}/{}" .format(docs_path, name))
                
                # copy docs
                os.system(
                    "cp -r {}/docs/{}/* {}/{}/" .format(work_path, language, docs_path, name))

                # add index
                os.system(
                    "echo '\r    {}/index' >> {}/index.rst" .format(name, docs_path))


if __name__ == '__main__':

    args = parse_args()

    config_file = os.path.abspath(args.config)
    docs_dir = os.path.abspath(args.dir)

    if not os.path.exists(config_file):
        print('Config file not found.')
        sys.exit(1)

    if not os.path.exists(docs_dir):
        print('Docs directory not found.')
        sys.exit(1)

    main(config_file, docs_dir)
