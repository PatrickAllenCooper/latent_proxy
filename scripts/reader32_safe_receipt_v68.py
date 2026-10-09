"""Strict output-only receipt writer; never accepts directory/distribution data."""
import hashlib
import json
import math
import os
from pathlib import Path
import re

MAX_BYTES = 3072
TRUE_KEYS = ('complete', 'CPU_only', 'guarded_imports_completed', 'source_hashes_valid')
FALSE_KEYS = ('qualification_established',)
ZERO_KEYS = ('model_loads', 'model_calls', 'GPU_allocations')
TIME_KEYS = ('started', 'completed')
HASH_KEYS = ('CPU_receipt_sha256', 'runtime_receipt_sha256', 'freeze_sha256',
             'baseline_driver_sha256', 'adapter_driver_sha256')
KEYS = set(TRUE_KEYS + FALSE_KEYS + ZERO_KEYS + TIME_KEYS + HASH_KEYS)


def encode_receipt(receipt, max_bytes=MAX_BYTES):
    if type(receipt) is not dict or set(receipt) != KEYS:
        raise ValueError('safe receipt schema rejected')
    for k in TRUE_KEYS:
        if receipt[k] is not True: raise ValueError('safe receipt boolean rejected')
    for k in FALSE_KEYS:
        if receipt[k] is not False: raise ValueError('safe receipt boolean rejected')
    for k in ZERO_KEYS:
        if type(receipt[k]) is not int or receipt[k] != 0: raise ValueError('safe receipt counter rejected')
    for k in TIME_KEYS:
        value=receipt[k]
        if type(value) not in (int,float) or not 0 <= value <= 1e12 or not math.isfinite(value):
            raise ValueError('safe receipt timestamp rejected')
    if receipt['completed'] < receipt['started']: raise ValueError('safe receipt time order rejected')
    for k in HASH_KEYS:
        if type(receipt[k]) is not str or not re.fullmatch('[0-9a-f]{64}',receipt[k]):
            raise ValueError('safe receipt hash rejected')
    data=(json.dumps(receipt,allow_nan=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    if not 0 < max_bytes <= MAX_BYTES or len(data)>max_bytes:
        raise ValueError('safe receipt byte cap')
    return data


def write_receipt(path, receipt, max_bytes=MAX_BYTES):
    # Validate and cap before opening: rejected fields never touch disk.
    data=encode_receipt(receipt,max_bytes)
    with Path(path).open('xb') as out:
        out.write(data);out.flush();os.fsync(out.fileno())
    directory=os.open(str(Path(path).parent),os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
