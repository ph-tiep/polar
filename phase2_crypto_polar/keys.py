"""Assumed shared simulation secret, with HMAC-SHA256 per-frame keystreams.

This does not implement reciprocal-channel key extraction or key distribution.
Keys stay in a private local file; manifests contain only a fingerprint/nonce.
"""
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import numpy as np
import torch


def load_run_key(output, key_file=None, nonce=None):
    if key_file is None:
        folder = Path(output)/'private'
        folder.mkdir(mode=0o700)
        path = folder/'shared_key.bin'
        secret = secrets.token_bytes(32)
        # Exclusive creation; never overwrite user material.
        with path.open('xb') as f:
            f.write(secret)
        path.chmod(0o600)
    else:
        path = Path(key_file)
        secret = path.read_bytes()
    if len(secret) != 32:
        raise ValueError('Shared key file must contain exactly 32 raw bytes')
    nonce = nonce or secrets.token_hex(16)
    try:
        nonce_bytes = bytes.fromhex(nonce)
    except (ValueError, TypeError) as exc:
        raise ValueError('Key nonce must be 32 hexadecimal characters') from exc
    if len(nonce_bytes) != 16:
        raise ValueError('Key nonce must be 32 hexadecimal characters')
    return secret, nonce_bytes, dict(
        assumption='Alice and Bob share a secret; no CIR key extraction or exchange is simulated',
        generator='HMAC-SHA256 counter PRF v1', key_bits=256,
        fingerprint_sha256=hashlib.sha256(secret).hexdigest(), nonce=nonce,
        local_key_file=str(path.resolve()),
        warning='Private simulation file: do not publish. Replay requires this file and the nonce.')


class SharedKeyGenerator:
    def __init__(self, master_key, nonce, *, n, k, crc):
        if len(master_key) != 32 or len(nonce) != 16:
            raise ValueError('Expected 32-byte shared key and 16-byte nonce')
        self.master_key = master_key
        # Kept as a public context attribute for paired perturbation diagnostics.
        self.csi_digest = nonce
        self.context = json.dumps(dict(n=n, k=k, crc=crc), sort_keys=True, separators=(',', ':')).encode()

    def generate_key_stream(self, batch_size, n, frame_idx=0, device=None, dtype=torch.float32):
        if batch_size < 1 or n < 1 or frame_idx < 0:
            raise ValueError('Invalid keystream dimensions/frame index')
        rows = []
        for frame in range(frame_idx, frame_idx+batch_size):
            prefix = (b'POLAR-SIM-KEYSTREAM-v1\0' + self.csi_digest + len(self.context).to_bytes(4, 'big') +
                      self.context + n.to_bytes(4, 'big') + frame.to_bytes(8, 'big'))
            size = (n+7)//8
            raw = b''.join(hmac.digest(self.master_key, prefix+counter.to_bytes(4, 'big'), 'sha256')
                           for counter in range((size+31)//32))[:size]
            rows.append(np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n])
        return torch.as_tensor(np.stack(rows), dtype=dtype, device=device)
