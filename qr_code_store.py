import json
import math
import gzip
import os.path
from tqdm import tqdm
import qrcode
from base64 import b85decode, b85encode
from hashlib import sha256

import typing

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


FILENAME_PREFIX = chunk_data(int_to_bytes(1))
FILENAME_SUFFIX = chunk_data(int_to_bytes(2))


def sha256_digest(data: bytes):
    return sha256(data, usedforsecurity=False).hexdigest()


def create_meta_chunk(chunk_index: int, **kwargs) -> bytes:
    json_text = json.dumps(dict(i=chunk_index, **kwargs))
    json_bytes = json_text.encode('utf-8')
    chunks = chunk_data(json_bytes)
    assert len(chunks) == 1
    return chunks[0]


def parse_meta_chunk(meta_chunk: bytes) -> dict:
    json_bytes = unchunk_data([meta_chunk])
    obj = json.loads(json_bytes.decode('utf-8'))
    return obj


def save_qr_code_to_file(data: bytes, file: str, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M):
    if os.path.isfile(file):
        return
    assert len(data) > 0, f'No data provided for QR code for "{file}"!'
    qr = qrcode.main.QRCode(version=qr_version, error_correction=error_correction)
    qr.add_data(data)
    image = qr.make_image(fill_color="black", back_color="white")
    image.save(file)


def generate_qr_codes(datas: typing.List[bytes], output_directory: str, qr_version: int = 40,
                      error_correction=qrcode.ERROR_CORRECT_M) -> typing.List[str]:
    files = []
    os.makedirs(output_directory, exist_ok=True)
    for i, data in enumerate(tqdm(datas, desc=f'Generating QR codes in directory "{output_directory}"')):
        file_name = os.path.join(output_directory, f'chunk_{i}.png')
        save_qr_code_to_file(data, file_name, qr_version=qr_version, error_correction=error_correction)
        files.append(file_name)
    return files


def store(data: bytes, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M, images_per_frame=2,
          color_channels: str = 'RGB', gzip_data: bool = True,
          output_directory: str = 'output', **additional_meta):
    encode = ''
    original_size = len(data)
    if gzip_data:
        data = gzip.compress(data)
        size_delta = len(data) - original_size
        print(
            f'Gzipped data size delta: {size_delta / 1024 / 1024:0.2f} MB, {(size_delta + original_size) / original_size * 100: 0.2f}%')
        encode += 'gzip,'
    encoded_data = b85encode(data)
    encode += 'b85'
    chunks = chunk_data(encoded_data, hide_progress=False)
    generate_qr_codes(chunks, output_directory, qr_version, error_correction)
    common_meta = dict(hash=sha256_digest(encoded_data), encode=encode, chunks=len(chunks), **additional_meta)
    qr_codes_per_frame = images_per_frame * len(color_channels)
    usable_qr_codes = qr_codes_per_frame - 1  # one used for meta
    meta_packaged_chunks = []
    total_chunk_packages = 0
    for i in range(0, len(chunks), usable_qr_codes):
        batch = chunks[i:i + usable_qr_codes]
        batch_hashes = list(map(sha256_digest, batch))
        meta_chunk = create_meta_chunk(i, **common_meta, batch_hashes=batch_hashes)
        packaged_chunk = [meta_chunk] + batch
        total_chunk_packages += len(packaged_chunk)
        meta_packaged_chunks.append(packaged_chunk)
    print(
        f'Number of frames: {len(meta_packaged_chunks)}. Approximate size in bytes: {total_chunk_packages * storage_limit / 1024 / 1024: 0.3f} MB')
    size_delta = (total_chunk_packages * storage_limit) - original_size
    print(
        f'Change in size from packaged chunking: {size_delta / 1024 / 1024: 0.2f} MB, {(size_delta + original_size) / original_size * 100: 0.2f}%')
    return meta_packaged_chunks


def store_file(file_path: str, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M,
               frames_per_second: int = 23, output_directory: str = 'output'):
    with open(file_path, 'rb') as f:
        data = f.read()
    file_name = os.path.split(file_path)[1]
    chunks = store(data, qr_version=qr_version, error_correction=error_correction, file=file_name,
                   output_directory=output_directory)
    # b85_encoded_data, decoded_file_name = decode_byte_chunks(chunks)
    # decoded_data = b85decode(b85_encoded_data)
    # assert file_name == decoded_file_name, f'File name prefix does not match: "{file_name}" != "{decoded_file_name}"'
    # assert len(data) == len(
    #     decoded_data), f'Decoded data length does not match input data: {len(data)} != {len(decoded_data)}'
    # assert data == decoded_data, 'Data does not match encoded data'
    # number_qr_codes = math.ceil(len(chunks) / 3)
    # print(f'Estimated number of QR codes: {number_qr_codes}')
    # frames = math.ceil(number_qr_codes / qr_codes_per_frame)
    # print(f'Estimated number of frames: {frames}')
    # estimated_seconds = math.ceil(frames / frames_per_second)
    # print(f'Estimated seconds of video: {estimated_seconds}. Minutes: {estimated_seconds / 60:0.2f}')


if __name__ == '__main__':
    store_file('test_file.txt')
