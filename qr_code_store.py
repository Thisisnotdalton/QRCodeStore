#!/usr/bin/env python
import json
import math
import gzip
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import os.path

from PIL.Image import Resampling
from tqdm import tqdm
from base64 import b85decode, b85encode
from hashlib import sha256
import typing

import qrcode
import numpy as np
from PIL import Image

error_corrections = {
    'l': qrcode.ERROR_CORRECT_L,
    'm': qrcode.ERROR_CORRECT_M,
    'q': qrcode.ERROR_CORRECT_Q,
    'h': qrcode.ERROR_CORRECT_H,
}

error_correction_names = {
    v: k for k, v in error_corrections.items()
}


def _lmqh_qr_limits(l, m, q, h):
    return {
        qrcode.ERROR_CORRECT_L: l,
        qrcode.ERROR_CORRECT_M: m,
        qrcode.ERROR_CORRECT_Q: q,
        qrcode.ERROR_CORRECT_H: h
    }



qr_code_sizes = {
    1: 21,
    10: 57,
    11: 61,
    20: 97,
    21: 101,
    30: 137,
    31: 141,
    40: 177
}


def normalize_error_correction(error_correction) -> int:
    if isinstance(error_correction, str):
        if error_correction.lower() not in error_corrections:
            print(f'Could not parse error correction {error_correction}. Defaulting to Q.')
            error_correction = qrcode.ERROR_CORRECT_Q
        else:
            error_correction = error_corrections[error_correction.lower()]
    return error_correction


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_storage_limit(qr_version: int, error_correction: typing.Union[int, str]):
    min_limit = 1
    max_limit = 2953
    # storage limits taken from https://www.qrcode.com/en/about/version.html
    data = ('a' * max_limit).encode('utf-8')
    print(f'Evaluating limits for QR v{qr_version} with error correction: {error_correction}')
    while min_limit < max_limit:
        actual_limit = int((min_limit + 1 + max_limit) / 2)
        try:
            save_qr_code(data[:actual_limit], None, qr_version, error_correction)
            min_limit = actual_limit
        except Exception as e:
            max_limit = actual_limit - 1
    actual_limit = min_limit
    print(f'Effective limit for QR v{qr_version} with error correction'
          f' "{error_correction_names[error_correction]}" -> {actual_limit}')
    return actual_limit


def bytes_to_represent_int(x: int) -> int:
    return math.ceil(math.log2(max(2, x)) / 8)


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(bytes_to_represent_int(x))


def chunk_data(data: bytes, chunk_size: int, hide_progress: bool = True) -> typing.List[bytes]:
    chunk_size_bytes = int_to_bytes(chunk_size)
    encoded_chunk_size = b85encode(chunk_size_bytes)
    bytes_for_chunk_length = len(chunk_size_bytes)
    usable_chunk_size = chunk_size - len(encoded_chunk_size)
    chunks = []
    for i in tqdm(range(0, len(data), usable_chunk_size), desc='Chunking data', disable=hide_progress):
        chunk_end = min(i + usable_chunk_size, len(data))
        data_chunk = data[i:chunk_end]
        assert len(data_chunk) <= usable_chunk_size
        chunk_length_bytes = int(chunk_end - i).to_bytes(bytes_for_chunk_length)
        length_prefix = b85encode(chunk_length_bytes)
        chunk = length_prefix + data_chunk
        assert len(chunk) <= chunk_size
        chunks.append(chunk)
    return chunks


def unchunk_data(chunks: typing.List[bytes], chunk_size: int, hide_progress: bool = True) -> bytes:
    data = bytes()
    bytes_for_chunk_length = bytes_to_represent_int(chunk_size)
    usable_chunk_size = chunk_size - bytes_for_chunk_length
    for chunk in tqdm(chunks, desc='Combining data chunks', disable=hide_progress):
        assert len(chunk) == chunk_size
        chunk_length = int.from_bytes(chunk[:bytes_for_chunk_length])
        assert chunk_length <= usable_chunk_size, f'Chunk has length exceeding usable chunk length: {chunk_length} > {usable_chunk_size}'
        data += chunk[bytes_for_chunk_length:bytes_for_chunk_length + chunk_length]
    return data


def sha256_digest(data: bytes):
    return sha256(data, usedforsecurity=False).hexdigest()


def create_json_chunk(chunks: dict, **common_meta):
    json_text = json.dumps(dict(chunks=chunks, **common_meta), separators=(',', ':'), sort_keys=True)
    chunk = json_text.encode('utf-8')
    return chunk


def create_meta_chunks(chunkable_meta: typing.Dict[int, str], storage_limit: int, **common_meta) -> typing.List[bytes]:
    chunks = []
    max_key_length = max(list(map(len, map(str, chunkable_meta.keys()))))
    max_value_length = max(list(map(len, chunkable_meta.values())))
    max_example_item = ['K' * max_key_length, 'V' * max_value_length]
    max_chunk_size = storage_limit
    min_chunk_size = 1
    while max_chunk_size > min_chunk_size:
        chunk_size = int((min_chunk_size + max_chunk_size) / 2)
        if chunk_size == min_chunk_size:
            chunk_size += 1
        example_chunks = [max_example_item for i in range(chunk_size)]
        chunk = create_json_chunk(example_chunks, **common_meta)
        if len(chunk) > storage_limit:
            max_chunk_size = chunk_size - 1
        else:
            min_chunk_size = chunk_size
    print(f'Max chunk size: {min_chunk_size}')
    queue = list(sorted(chunkable_meta.keys()))
    while len(queue) > 0:
        batch = queue[:min_chunk_size]
        queue = queue[min_chunk_size:]
        batch_chunk = {
            str(k): chunkable_meta[k] for k in batch
        }
        meta_chunk = create_json_chunk(batch_chunk, **common_meta)
        assert len(meta_chunk) <= storage_limit
        chunks.append(meta_chunk)
    return chunks


def parse_meta_chunk(meta_chunk: bytes, chunk_size: int) -> dict:
    json_bytes = unchunk_data([meta_chunk], chunk_size=chunk_size)
    obj = json.loads(json_bytes.decode('utf-8'))
    return obj


def save_qr_code(data: bytes, file: str, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M):
    assert len(data) > 0, f'No data provided for QR code for "{file}"!'
    make_code = file is None or not os.path.isfile(file)
    if make_code:
        qr = qrcode.main.QRCode(version=qr_version, error_correction=error_correction, box_size=1)
        qr.add_data(data)
        image = qr.make_image(fill_color="black", back_color="white")
        if file is None:
            return image
        image.save(file)
    return file


def _concurrent_save_qr(args):
    return save_qr_code(*args)


def generate_qr_codes(datas: typing.Dict[str, bytes], qr_version: int = 40,
                      error_correction=qrcode.ERROR_CORRECT_M):
    print(f'QR version: {qr_version}. Error correction: {error_correction_names[error_correction]}.')
    with ProcessPoolExecutor() as executor:
        arguments = [
            [data, file, qr_version, error_correction]
            for file, data in datas.items()
        ]
        futures = executor.map(_concurrent_save_qr, arguments, chunksize=1)
        for future in tqdm(futures, total=len(arguments),
                           desc=f'Generating QR codes'):
            pass


def combine_qr_code(out_file: str, *channels):
    if os.path.isfile(out_file):
        return
    output_array = None
    for i, channel_file in enumerate(channels):
        with Image.open(channel_file) as channel_image:
            if i == 0:
                output_array = np.zeros((*channel_image.size, 3), dtype=np.uint8)
            else:
                channel_image = channel_image.rotate(90 * i, expand=False, resample=Resampling.NEAREST)
            output_array[:, :, i] = np.asarray(channel_image) * 255
    output_image = Image.fromarray(output_array)
    output_image.save(out_file)


def _concurrent_combine_qr_codes(args):
    combine_qr_code(*args)


def combine_qr_code_files(qr_code_files: typing.List[str], output_directory: str) -> typing.List[str]:
    queue = list(qr_code_files)
    files = []
    jobs = []
    digits_for_enumeration = len(str(math.ceil(len(qr_code_files) / 3)))
    os.makedirs(output_directory, exist_ok=True)
    while len(queue) > 0:
        batch = queue[:3]
        queue = queue[3:]
        batch_hash = sha256_digest(str(batch).encode('utf-8'))
        file_name = os.path.join(output_directory, f'{len(files):0{digits_for_enumeration}}_{batch_hash}.png')
        files.append(file_name)
        jobs.append([file_name, *batch])
    with ProcessPoolExecutor() as executor:
        futures = executor.map(_concurrent_combine_qr_codes, jobs, chunksize=1)
        for future in tqdm(futures, total=len(jobs), desc=f'Combining QR codes'):
            pass
    return files


def store(data: bytes, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M, gzip_data: bool = True,
          output_directory: str = 'output', target_framerate: int = 24, **additional_meta):
    encode = ''
    original_size = len(data)
    if gzip_data:
        data = gzip.compress(data, mtime=0)
        size_delta = len(data) - original_size
        print(
            f'Gzipped data size delta: {size_delta / 1024 / 1024:0.2f} MB, {(size_delta + original_size) / original_size * 100: 0.2f}%')
        encode += 'gzip,'
    error_correction = normalize_error_correction(error_correction)
    chunk_size = get_storage_limit(qr_version, error_correction)
    print(f'Using QR size of {chunk_size}')
    chunks = chunk_data(data, chunk_size=chunk_size, hide_progress=False)
    chunk_hashes = {
        i: sha256_digest(chunk)
        for i, chunk in enumerate(chunks)
    }
    digest_to_chunk = {
        digest: chunks[i]
        for i, digest in chunk_hashes.items()
    }
    chunk_directory = os.path.join(output_directory, 'chunks')
    os.makedirs(chunk_directory, exist_ok=True)
    chunk_file_paths = {
        os.path.join(chunk_directory, f'{digest}.png'): chunk
        for digest, chunk in digest_to_chunk.items()
    }
    generate_qr_codes(chunk_file_paths, qr_version, error_correction)
    common_meta = dict(hash=sha256_digest(data), encode=encode, chunkCount=len(chunks), **additional_meta)
    meta_chunks = create_meta_chunks(chunk_hashes, storage_limit=chunk_size, **common_meta)
    digits_for_enumeration = len(str(len(meta_chunks)))
    meta_digest_to_chunk = {
        f'{i:0{digits_for_enumeration}}_{sha256_digest(meta_chunk)}': meta_chunk
        for i, meta_chunk in enumerate(meta_chunks)
    }
    digest_to_chunk.update(meta_digest_to_chunk)
    assert len(digest_to_chunk) == len(chunks) + len(meta_chunks), f'Hash collision detected!'
    meta_directory = os.path.join(output_directory, 'meta')
    os.makedirs(meta_directory, exist_ok=True)
    meta_chunk_file_paths = {
        os.path.join(meta_directory, f'{digest}.png'): chunk
        for digest, chunk in meta_digest_to_chunk.items()
    }

    generate_qr_codes(meta_chunk_file_paths, qr_version, error_correction)
    all_qr_code_files = sorted(meta_chunk_file_paths.keys()) + sorted(chunk_file_paths.keys())
    colored_image_files = combine_qr_code_files(all_qr_code_files, os.path.join(output_directory, 'combined'))
    import random
    random.seed(data)
    colored_images = list(
        map(lambda _i: Image.open(colored_image_files[_i]).rotate((random.randint(0, 4) * 90) % 360, expand=False,
                                                                  resample=Resampling.NEAREST),
            range(len(colored_image_files))))
    sequence_image = colored_images.pop(0)
    common_sequence_config = dict(
        append_images=colored_images,
        loop=0,
        save_all=True,
        duration=1000 / target_framerate
    )
    sequences = {
        os.path.join(output_directory, 'sequence.gif'): dict(disposal=2, **common_sequence_config),
        os.path.join(output_directory, 'sequence.webp'): dict(lossless=True, method=6, **common_sequence_config)
    }

    for file_path, arguments in sequences.items():
        if not os.path.exists(file_path):
            sequence_image.save(file_path, **arguments)
        file_size = os.path.getsize(file_path)
        file_size_ratio = file_size / original_size * 100
        print(f'Sequence image saved at {file_path} with file size: {sizeof_fmt(file_size)}.'
              f' ({file_size_ratio:0.2f}% of original size,'
              f' {sizeof_fmt(file_size-original_size)} larger)')


def store_file(file_path: str, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M,
               output_directory: str = 'output'):
    with open(file_path, 'rb') as f:
        data = f.read()
    print(f'File sha256: {sha256_digest(data)}')
    file_name = os.path.split(file_path)[1]
    store(data, qr_version=qr_version, error_correction=error_correction,
          file=file_name, output_directory=output_directory)


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('file',
                                 help='File path of the file to store in an RGB QR code sequence')
    argument_parser.add_argument('-o', '--output-directory',
                                 default='output', help='Directory to store output files')
    argument_parser.add_argument('-q', '--qr_version', default=40, help='QR code version to use')
    argument_parser.add_argument('-e', '--error-correction', default="M",
                                 help='Error correction to use. One of H, Q, M, or L. Defaults to M.')
    args = argument_parser.parse_args()
    store_file(args.file, args.qr_version, args.error_correction, args.output_directory)


if __name__ == '__main__':
    main()
