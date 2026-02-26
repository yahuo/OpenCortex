import hashlib
import hmac
import os
import struct
import tempfile
import unittest
from pathlib import Path

import decrypt_wechat_db as dec


def _make_plaintext_db_page(fill_byte: int = 0x41) -> bytes:
    page = bytearray(dec.PAGE_SIZE)
    page[: len(dec.SQLITE_HEADER)] = dec.SQLITE_HEADER
    page[len(dec.SQLITE_HEADER) : dec.PAGE_SIZE - dec.RESERVE] = bytes([fill_byte]) * (
        dec.PAGE_SIZE - dec.RESERVE - len(dec.SQLITE_HEADER)
    )
    return bytes(page)


def _encrypt_v4_pages(key_hex: str, pages: list[bytes]) -> bytes:
    raw_key = dec.decode_hex_key(key_hex)
    salt = bytes(range(1, dec.SALT_SIZE + 1))
    enc_key, mac_key = dec.derive_keys(raw_key, salt)

    result = bytearray()
    for page_index, plain_page in enumerate(pages):
        if len(plain_page) != dec.PAGE_SIZE:
            raise ValueError("plain page size mismatch")

        offset = dec.SALT_SIZE if page_index == 0 else 0
        iv = bytes(((page_index + i) % 256) for i in range(dec.IV_SIZE))

        encrypted_section = plain_page[offset : dec.PAGE_SIZE - dec.RESERVE]
        ciphertext = dec._aes_cbc_encrypt(enc_key, iv, encrypted_section)

        page = bytearray(dec.PAGE_SIZE)
        if page_index == 0:
            page[: dec.SALT_SIZE] = salt
            page[dec.SALT_SIZE : dec.PAGE_SIZE - dec.RESERVE] = ciphertext
        else:
            page[: dec.PAGE_SIZE - dec.RESERVE] = ciphertext

        iv_start = dec.PAGE_SIZE - dec.RESERVE
        data_end = dec.PAGE_SIZE - dec.RESERVE + dec.IV_SIZE
        page[iv_start:data_end] = iv

        mac = hmac.new(mac_key, digestmod=hashlib.sha512)
        mac.update(page[offset:data_end])
        mac.update(struct.pack("<I", page_index + 1))
        page[data_end : data_end + dec.HMAC_SIZE] = mac.digest()

        result.extend(page)

    return bytes(result)


class TestDecryptWechatDb(unittest.TestCase):
    def test_decode_hex_key_rejects_non_hex(self) -> None:
        with self.assertRaisesRegex(dec.DecryptError, "decode hex"):
            dec.decode_hex_key("not-a-hex-key")

    def test_decode_hex_key_rejects_wrong_size(self) -> None:
        with self.assertRaisesRegex(dec.DecryptError, "32 bytes"):
            dec.decode_hex_key("ab" * 31)

    def test_decrypt_db_file_copies_plaintext_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "plain.db")
            dst = os.path.join(tmp, "out", "plain.db")
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            plain = _make_plaintext_db_page(fill_byte=0x42)
            with open(src, "wb") as fp:
                fp.write(plain)

            mode = dec.decrypt_db_file(src, dst, "11" * 32)
            self.assertEqual(mode, "copied")
            self.assertEqual(Path(src).read_bytes(), Path(dst).read_bytes())

    def test_decrypt_db_file_rejects_wrong_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "encrypted.db")
            dst = os.path.join(tmp, "out", "encrypted.db")
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            encrypted = _encrypt_v4_pages("11" * 32, [_make_plaintext_db_page()])
            Path(src).write_bytes(encrypted)

            with self.assertRaisesRegex(dec.DecryptError, "incorrect decryption key"):
                dec.decrypt_db_file(src, dst, "22" * 32)

    def test_decrypt_db_file_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "encrypted.db")
            dst = os.path.join(tmp, "out", "encrypted.db")
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            encrypted = _encrypt_v4_pages("33" * 32, [_make_plaintext_db_page(fill_byte=0x7A)])
            Path(src).write_bytes(encrypted)

            mode = dec.decrypt_db_file(src, dst, "33" * 32)
            self.assertEqual(mode, "decrypted")

            out = Path(dst).read_bytes()
            self.assertEqual(out[: len(dec.SQLITE_HEADER)], dec.SQLITE_HEADER)
            self.assertEqual(out[len(dec.SQLITE_HEADER)], 0x7A)

    def test_batch_decrypt_skips_fts_and_continues_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "src")
            work_dir = os.path.join(tmp, "out")

            plain_db = os.path.join(data_dir, "db_storage", "message", "message_0.db")
            fts_db = os.path.join(data_dir, "db_storage", "fts", "search.db")
            bad_db = os.path.join(data_dir, "db_storage", "message", "bad.db")

            os.makedirs(os.path.dirname(plain_db), exist_ok=True)
            os.makedirs(os.path.dirname(fts_db), exist_ok=True)

            Path(plain_db).write_bytes(_make_plaintext_db_page(fill_byte=0x31))
            Path(fts_db).write_bytes(_make_plaintext_db_page(fill_byte=0x32))
            Path(bad_db).write_bytes(_encrypt_v4_pages("aa" * 32, [_make_plaintext_db_page(fill_byte=0x33)]))

            stats, failures = dec.batch_decrypt(data_dir, work_dir, "bb" * 32)

            self.assertEqual(stats.total, 2)  # fts file excluded
            self.assertEqual(stats.success, 1)
            self.assertEqual(stats.failed, 1)
            self.assertEqual(len(failures), 1)
            self.assertIn("bad.db", failures[0].db_path)

            copied = os.path.join(work_dir, "db_storage", "message", "message_0.db")
            self.assertTrue(os.path.exists(copied))

            skipped_fts_out = os.path.join(work_dir, "db_storage", "fts", "search.db")
            self.assertFalse(os.path.exists(skipped_fts_out))

    def test_batch_decrypt_rejects_same_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "src")
            os.makedirs(data_dir, exist_ok=True)
            db_file = os.path.join(data_dir, "a.db")
            Path(db_file).write_bytes(_make_plaintext_db_page())

            with self.assertRaisesRegex(dec.DecryptError, "unsafe output directory"):
                dec.batch_decrypt(data_dir, data_dir, "11" * 32)

    def test_batch_decrypt_rejects_nested_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "src")
            work_dir = os.path.join(data_dir, "out")
            os.makedirs(data_dir, exist_ok=True)
            db_file = os.path.join(data_dir, "a.db")
            Path(db_file).write_bytes(_make_plaintext_db_page())

            with self.assertRaisesRegex(dec.DecryptError, "unsafe output directory"):
                dec.batch_decrypt(data_dir, work_dir, "11" * 32)

    def test_batch_decrypt_rejects_parent_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "src")
            os.makedirs(data_dir, exist_ok=True)
            db_file = os.path.join(data_dir, "a.db")
            Path(db_file).write_bytes(_make_plaintext_db_page())

            with self.assertRaisesRegex(dec.DecryptError, "unsafe output directory"):
                dec.batch_decrypt(data_dir, tmp, "11" * 32)

if __name__ == "__main__":
    unittest.main()
