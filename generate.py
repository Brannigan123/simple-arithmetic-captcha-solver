#!/usr/bin/env python3

import json
import os
import shlex
import cairosvg
from subprocess import Popen, PIPE, STDOUT

from const import ROOT_FOLDER, TEST_DATA_PATH, TRAIN_DATA_PATH


def generate_svg():
    cmd = shlex.split(f'node {ROOT_FOLDER}/generate.js')
    process = Popen(cmd, stdin=PIPE, stdout=PIPE,
                    stderr=STDOUT, close_fds=True)
    return json.loads(process.stdout.read())


def generate_png(dir, id):
    gen_svg = generate_svg()
    cairosvg.svg2png(
        gen_svg['data'], write_to=f'{dir}/{id} [{gen_svg["text"]}].png')


def generate_dataset(size=10000):
    os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
    os.makedirs(TEST_DATA_PATH, exist_ok=True)
    train_size = int(0.7 * size)
    for i in range(size):
        generate_png(TRAIN_DATA_PATH if i < train_size else TEST_DATA_PATH, i)


if __name__ == '__main__':
    generate_dataset()
