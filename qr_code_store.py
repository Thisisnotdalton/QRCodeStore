import json
import math
import gzip
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

storage_limit = 1852


def bytes_to_represent_int(x: int) -> int:
    return math.ceil(math.log2(max(2, x)) / 8)


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(bytes_to_represent_int(x))


def chunk_data(data: bytes, chunk_size: int = storage_limit, hide_progress: bool = True) -> typing.List[bytes]:
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


def unchunk_data(chunks: typing.List[bytes], chunk_size: int = storage_limit, hide_progress: bool = True) -> bytes:
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


def create_meta_chunks(chunkable_meta: typing.Dict[int, str], max_chunk_size: int = 1024, min_chunk_size: int = 1,
                       **common_meta) -> \
        typing.List[bytes]:
    chunks = []
    max_key_length = max(list(map(len, map(str, chunkable_meta.keys()))))
    max_value_length = max(list(map(len, chunkable_meta.values())))
    max_example_item = ['K' * max_key_length, 'V' * max_value_length]
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


def parse_meta_chunk(meta_chunk: bytes) -> dict:
    json_bytes = unchunk_data([meta_chunk])
    obj = json.loads(json_bytes.decode('utf-8'))
    return obj


def save_qr_code_to_file(data: bytes, file: str, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M):
    assert len(data) > 0, f'No data provided for QR code for "{file}"!'
    if not os.path.isfile(file):
        qr = qrcode.main.QRCode(version=qr_version, error_correction=error_correction, box_size=1)
        qr.add_data(data)
        image = qr.make_image(fill_color="black", back_color="white")
        image.save(file)
    return file


def _concurrent_save_qr(args):
    return save_qr_code_to_file(*args)


def generate_qr_codes(datas: typing.Dict[str, bytes], qr_version: int = 40,
                      error_correction=qrcode.ERROR_CORRECT_M):
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
    encoded_data = b85encode(data)
    encode += 'b85'
    chunks = chunk_data(encoded_data, hide_progress=False)
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
    common_meta = dict(hash=sha256_digest(encoded_data), encode=encode, chunkCount=len(chunks), **additional_meta)
    meta_chunks = create_meta_chunks(chunk_hashes, **common_meta)
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
    random.seed(encoded_data)
    colored_images = list(map(lambda _i: Image.open(colored_image_files[_i]).rotate((random.randint(0,4) * 90) % 360, expand=False, resample=Resampling.NEAREST),
                              range(len(colored_image_files))))
    sequence_image = colored_images.pop(0)
    sequence_image.save(os.path.join(output_directory, 'sequence.gif'), save_all=True, disposal=2,
                        append_images=colored_images, loop=0, duration=1000/target_framerate)


def store_file(file_path: str, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M,
               output_directory: str = 'output'):
    with open(file_path, 'rb') as f:
        data = f.read()
    print(f'File sha256: {sha256_digest(data)}')
    file_name = os.path.split(file_path)[1]
    chunks = store(data, qr_version=qr_version, error_correction=error_correction, file=file_name,
                   output_directory=output_directory)


if __name__ == '__main__':
    store_file('test_file.txt')
