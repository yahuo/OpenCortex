#!/usr/bin/env python3
"""Decrypt WeChat local SQLCipher DBs for macOS WeChat v4."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple

try:
    from Crypto.Cipher import AES as CryptoAES
except ImportError:  # offline/local environments may not have pycryptodome installed yet
    CryptoAES = None

SQLITE_HEADER = b"SQLite format 3\x00"

KEY_SIZE = 32
SALT_SIZE = 16
IV_SIZE = 16
AES_BLOCK_SIZE = 16

PAGE_SIZE = 4096
ITER_COUNT = 256000
HMAC_SIZE = 64
RESERVE = IV_SIZE + HMAC_SIZE  # 80, already aligned to AES block size.


class DecryptError(Exception):
    """Raised when decrypting a DB fails."""


@dataclass
class BatchStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class BatchFailure:
    db_path: str
    reason: str


def _normalize_dir(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _path_contains(parent: str, child: str) -> bool:
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:
        # Different drives/mount roots; treat as non-overlapping.
        return False


def ensure_safe_output_dir(data_dir: str, work_dir: str) -> None:
    data_norm = _normalize_dir(data_dir)
    work_norm = _normalize_dir(work_dir)

    if _path_contains(data_norm, work_norm) or _path_contains(work_norm, data_norm):
        raise DecryptError(
            "unsafe output directory: output-dir/work-dir must not be the same as data-dir "
            "or a parent/child path of it"
        )


def _aes_cbc_decrypt(key: bytes, iv: bytes, encrypted: bytes) -> bytes:
    if CryptoAES is not None:
        return CryptoAES.new(key, CryptoAES.MODE_CBC, iv=iv).decrypt(encrypted)

    proc = subprocess.run(
        [
            "openssl",
            "enc",
            "-d",
            "-aes-256-cbc",
            "-K",
            key.hex(),
            "-iv",
            iv.hex(),
            "-nosalt",
            "-nopad",
        ],
        input=encrypted,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise DecryptError(f"aes decrypt failed: {stderr or 'openssl error'}")
    return proc.stdout


def _aes_cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    if CryptoAES is not None:
        return CryptoAES.new(key, CryptoAES.MODE_CBC, iv=iv).encrypt(plaintext)

    proc = subprocess.run(
        [
            "openssl",
            "enc",
            "-aes-256-cbc",
            "-K",
            key.hex(),
            "-iv",
            iv.hex(),
            "-nosalt",
            "-nopad",
        ],
        input=plaintext,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise DecryptError(f"aes encrypt failed: {stderr or 'openssl error'}")
    return proc.stdout


def decode_hex_key(hex_key: str) -> bytes:
    try:
        key = bytes.fromhex(hex_key)
    except ValueError as exc:
        raise DecryptError("failed to decode hex key") from exc

    if len(key) != KEY_SIZE:
        raise DecryptError("key length must be 32 bytes (64 hex chars)")

    return key


def xor_bytes(data: bytes, value: int) -> bytes:
    return bytes(item ^ value for item in data)


def derive_keys(raw_key: bytes, salt: bytes) -> Tuple[bytes, bytes]:
    enc_key = hashlib.pbkdf2_hmac("sha512", raw_key, salt, ITER_COUNT, KEY_SIZE)
    mac_salt = xor_bytes(salt, 0x3A)
    mac_key = hashlib.pbkdf2_hmac("sha512", enc_key, mac_salt, 2, KEY_SIZE)
    return enc_key, mac_key


def _hmac_digest(mac_key: bytes, data: bytes, page_no: int) -> bytes:
    mac = hmac.new(mac_key, digestmod=hashlib.sha512)
    mac.update(data)
    mac.update(struct.pack("<I", page_no))
    return mac.digest()


def validate_key(page1: bytes, raw_key: bytes) -> bool:
    if len(page1) < PAGE_SIZE or len(raw_key) != KEY_SIZE:
        return False

    salt = page1[:SALT_SIZE]
    _, mac_key = derive_keys(raw_key, salt)

    data_end = PAGE_SIZE - RESERVE + IV_SIZE
    calculated = _hmac_digest(mac_key, page1[SALT_SIZE:data_end], 1)
    stored = page1[data_end : data_end + HMAC_SIZE]

    return hmac.compare_digest(calculated, stored)


def decrypt_page(page: bytes, enc_key: bytes, mac_key: bytes, page_index: int) -> bytes:
    offset = SALT_SIZE if page_index == 0 else 0

    data_end = PAGE_SIZE - RESERVE + IV_SIZE
    calculated = _hmac_digest(mac_key, page[offset:data_end], page_index + 1)
    stored = page[data_end : data_end + HMAC_SIZE]

    if not hmac.compare_digest(calculated, stored):
        raise DecryptError("hash verification failed during decryption")

    iv_start = PAGE_SIZE - RESERVE
    iv_end = iv_start + IV_SIZE
    iv = page[iv_start:iv_end]

    encrypted = page[offset : PAGE_SIZE - RESERVE]
    plaintext = _aes_cbc_decrypt(enc_key, iv, encrypted)

    return plaintext + page[PAGE_SIZE - RESERVE : PAGE_SIZE]


def _copy_as_is(src: str, dst: str) -> str:
    tmp = dst + ".tmp"
    try:
        with open(src, "rb") as src_fp, open(tmp, "wb") as dst_fp:
            shutil.copyfileobj(src_fp, dst_fp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return "copied"


def decrypt_db_file(src: str, dst: str, hex_key: str) -> str:
    raw_key = decode_hex_key(hex_key)

    with open(src, "rb") as db_fp:
        page1 = db_fp.read(PAGE_SIZE)

    if len(page1) < PAGE_SIZE:
        raise DecryptError("database file too small")

    if page1.startswith(SQLITE_HEADER):
        return _copy_as_is(src, dst)

    if not validate_key(page1, raw_key):
        raise DecryptError("incorrect decryption key or corrupted database")

    salt = page1[:SALT_SIZE]
    enc_key, mac_key = derive_keys(raw_key, salt)

    tmp = dst + ".tmp"
    try:
        with open(src, "rb") as src_fp, open(tmp, "wb") as out_fp:
            out_fp.write(SQLITE_HEADER)

            page_index = 0
            while True:
                page = src_fp.read(PAGE_SIZE)
                if not page:
                    break
                if len(page) < PAGE_SIZE:
                    break

                if all(byte == 0 for byte in page):
                    out_fp.write(page)
                    page_index += 1
                    continue

                out_fp.write(decrypt_page(page, enc_key, mac_key, page_index))
                page_index += 1

        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return "decrypted"


def list_db_files(data_dir: str) -> List[str]:
    db_files: List[str] = []

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if not filename.endswith(".db"):
                continue

            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, data_dir)
            if "fts" in rel_path:
                continue

            db_files.append(abs_path)

    db_files.sort()
    return db_files


def batch_decrypt(data_dir: str, work_dir: str, hex_key: str) -> Tuple[BatchStats, List[BatchFailure]]:
    if not os.path.isdir(data_dir):
        raise DecryptError(f"data directory does not exist: {data_dir}")

    ensure_safe_output_dir(data_dir, work_dir)

    db_files = list_db_files(data_dir)

    stats = BatchStats(total=len(db_files))
    failures: List[BatchFailure] = []

    for src in db_files:
        rel_path = os.path.relpath(src, data_dir)
        dst = os.path.join(work_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.realpath(src) == os.path.realpath(dst):
            stats.failed += 1
            failures.append(BatchFailure(db_path=rel_path, reason="unsafe path: source and destination are identical"))
            continue

        try:
            decrypt_db_file(src, dst, hex_key)
            stats.success += 1
        except Exception as exc:  # keep parity with Go behavior: continue on single-file failure
            stats.failed += 1
            failures.append(BatchFailure(db_path=rel_path, reason=str(exc)))

    return stats, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WeChat DB decrypt utility (Python)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decrypt_parser = subparsers.add_parser("decrypt", help="decrypt WeChat DB files")
    decrypt_parser.add_argument("-k", "--key", required=True, help="64-char hex key")
    decrypt_parser.add_argument("-d", "--data-dir", required=True, help="WeChat account data dir")
    decrypt_parser.add_argument(
        "-o",
        "-w",
        "--output-dir",
        "--work-dir",
        dest="work_dir",
        required=True,
        help="output work dir for decrypted db files",
    )
    decrypt_parser.add_argument("-p", "--platform", default="darwin", help="platform (default: darwin)")
    decrypt_parser.add_argument("-v", "--version", type=int, default=4, help="wechat db version (default: 4)")

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "decrypt":
        parser.error("unsupported command")

    if args.platform != "darwin" or args.version != 4:
        print(
            "error: this python implementation currently supports only platform=darwin, version=4",
            file=sys.stderr,
        )
        return 2

    try:
        stats, failures = batch_decrypt(args.data_dir, args.work_dir, args.key)
    except Exception as exc:
        print(f"decrypt failed: {exc}", file=sys.stderr)
        return 1

    print(f"decrypt completed: total={stats.total}, success={stats.success}, failed={stats.failed}")
    for failure in failures:
        print(f"failed: {failure.db_path} -> {failure.reason}")

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
