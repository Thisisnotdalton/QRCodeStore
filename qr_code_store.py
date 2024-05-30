import math
import os.path
from tqdm import tqdm
import qrcode
from base64 import b85decode, b85encode

import typing

storage_limit = 1852


def bytes_to_represent_int(x: int) -> int:
    return math.ceil(math.log2(max(2, x)) / 8)


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(bytes_to_represent_int(x))


def chunk_data(data: bytes, chunk_size: int = storage_limit) -> typing.List[bytes]:
    bytes_for_chunk_length = bytes_to_represent_int(chunk_size)
    usable_chunk_size = chunk_size - bytes_for_chunk_length
    chunks = []
    for i in tqdm(range(0, len(data), usable_chunk_size), desc='Chunking data'):
        chunk_end = min(i + usable_chunk_size, len(data))
        data_chunk = data[i:chunk_end]
        if len(data_chunk) < usable_chunk_size:
            data_chunk += bytes(usable_chunk_size - len(data_chunk))
        assert len(data_chunk) == usable_chunk_size
        length_prefix = int(chunk_end - i).to_bytes(bytes_for_chunk_length)
        chunk = length_prefix + data_chunk
        assert len(chunk) == chunk_size
        chunks.append(chunk)
    return chunks


def unchunk_data(chunks: typing.List[bytes], chunk_size: int = storage_limit) -> bytes:
    data = bytes()
    bytes_for_chunk_length = bytes_to_represent_int(chunk_size)
    usable_chunk_size = chunk_size - bytes_for_chunk_length
    for chunk in tqdm(chunks, desc='Combining data chunks'):
        assert len(chunk) == chunk_size
        chunk_length = int.from_bytes(chunk[:bytes_for_chunk_length])
        assert chunk_length <= usable_chunk_size, f'Chunk has length exceeding usable chunk length: {chunk_length} > {usable_chunk_size}'
        data += chunk[bytes_for_chunk_length:bytes_for_chunk_length + chunk_length]
    return data


FILENAME_PREFIX = chunk_data(int_to_bytes(1))
FILENAME_SUFFIX = chunk_data(int_to_bytes(2))


def store(data: bytes, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M, string_prefix: str = ''):
    encoded_data = b85encode(data)
    chunks = []
    if len(string_prefix) > 0:
        string_bytes = string_prefix.encode('utf-8')
        encoded_string_bytes = b85encode(string_bytes)
        chunks.extend(FILENAME_PREFIX)
        chunks.extend(chunk_data(encoded_string_bytes))
        chunks.extend(FILENAME_SUFFIX)
    chunks.extend(chunk_data(encoded_data))
    print(
        f'Number of chunks: {len(chunks)}. Approximate size in bytes: {len(chunks) * storage_limit / 1024 / 1024: 0.3f} MB')
    original_size = len(encoded_data)
    size_delta = (len(chunks) * storage_limit) - original_size
    print(f'Change in size from chunking: {size_delta} bytes, {(size_delta+original_size) / original_size * 100: 0.2f}%')
    return chunks


def decode_byte_chunks(chunks: typing.List[bytes]):
    file_name = None
    if chunks[:len(FILENAME_PREFIX)] == FILENAME_PREFIX:
        file_name_chunks = []
        chunks = chunks[len(FILENAME_PREFIX):]
        while chunks[:len(FILENAME_SUFFIX)] != FILENAME_SUFFIX:
            file_name_chunks.append(chunks.pop(0))
        chunks = chunks[len(FILENAME_SUFFIX):]
        file_name_bytes_encoded = decode_byte_chunks(file_name_chunks)[0]
        file_name_bytes = b85decode(file_name_bytes_encoded)
        file_name = file_name_bytes.decode('utf-8')
    return unchunk_data(chunks), file_name


def store_file(file_path: str, qr_version: int = 40, error_correction=qrcode.ERROR_CORRECT_M,
               frames_per_second: int = 23):
    with open(file_path, 'rb') as f:
        data = f.read()
    file_name = os.path.split(file_path)[1]
    chunks = store(data, qr_version=qr_version, error_correction=error_correction, string_prefix=file_name)
    b85_encoded_data, decoded_file_name = decode_byte_chunks(chunks)
    decoded_data = b85decode(b85_encoded_data)
    assert file_name == decoded_file_name, f'File name prefix does not match: "{file_name}" != "{decoded_file_name}"'
    assert len(data) == len(
        decoded_data), f'Decoded data length does not match input data: {len(data)} != {len(decoded_data)}'
    assert data == decoded_data, 'Data does not match encoded data'
    number_qr_codes = math.ceil(len(chunks) / 3)
    print(f'Estimated number of QR codes: {number_qr_codes}')
    qr_codes_per_frame = 2
    frames = math.ceil(number_qr_codes / qr_codes_per_frame)
    print(f'Estimated number of frames: {frames}')
    estimated_seconds = math.ceil(frames / frames_per_second)
    print(f'Estimated seconds of video: {estimated_seconds}. Minutes: {estimated_seconds / 60:0.2f}')


if __name__ == '__main__':
    store_file('test_file.txt')
