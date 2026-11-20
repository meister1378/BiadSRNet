"""Encrypted checkpoints (AES-256-GCM, key derived from a password with scrypt).

The key is read from the CKPT_KEY environment variable.
"""
import io
import os

import torch

MAGIC = b"ENCKPT1\0"
_N, _R, _P = 2 ** 15, 8, 1


def _key(password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    return Scrypt(salt=salt, length=32, n=_N, r=_R, p=_P).derive(password.encode("utf-8"))


def _password() -> str:
    pw = os.environ.get("CKPT_KEY")
    if not pw:
        raise RuntimeError("checkpoint is encrypted: set the CKPT_KEY environment variable")
    return pw


def encrypt_file(src: str, dst: str, password: str) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce = os.urandom(16), os.urandom(12)
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(MAGIC + salt + nonce + AESGCM(_key(password, salt)).encrypt(nonce, data, MAGIC))


def load_checkpoint(path: str, map_location="cpu"):
    """torch.load for plain or encrypted (.enc) checkpoints."""
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(MAGIC):
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        salt, nonce, body = data[8:24], data[24:36], data[36:]
        try:
            data = AESGCM(_key(_password(), salt)).decrypt(nonce, body, MAGIC)
        except InvalidTag:
            raise RuntimeError("wrong checkpoint key") from None
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)
