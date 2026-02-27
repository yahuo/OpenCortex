#!/usr/bin/env python3
import os
import sqlite3
import hashlib
import sys
import re
import glob
import logging
from datetime import datetime

try:
    import zstandard as zstd
except ImportError:
    zstd = None

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

WECHAT_XFILES_BASE = os.path.expanduser(
    "~/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files"
)


def detect_wechat_account_dirs() -> list[tuple[str, str]]:
    """
    扫描微信账号数据目录，返回所有已找到的账号目录。
    格式：[(显示名称, 完整路径), ...]
    一个有效的账号目录应该包含 db_storage/ 或 msg/ 子目录。
    """
    results = []
    if not os.path.isdir(WECHAT_XFILES_BASE):
        return results
    for entry in os.scandir(WECHAT_XFILES_BASE):
        if not entry.is_dir():
            continue
        # 判断是否是一个有效的账号目录
        has_db_storage = os.path.isdir(os.path.join(entry.path, "db_storage"))
        has_msg = os.path.isdir(os.path.join(entry.path, "msg"))
        if has_db_storage or has_msg:
            results.append((entry.name, entry.path))
    results.sort(key=lambda x: x[0])
    return results


_detected_dirs = detect_wechat_account_dirs()
DEFAULT_ROOT_BASE = _detected_dirs[0][1] if _detected_dirs else WECHAT_XFILES_BASE


def _auto_detect_db_dir() -> str:
    """自动探测已解密的数据库目录，优先查找 db_storage 子目录"""
    base = os.path.expanduser("~/wechat_db_backup")
    db_storage = os.path.join(base, "db_storage")
    contact_in_storage = os.path.join(db_storage, "contact", "contact.db")
    contact_in_base = os.path.join(base, "contact", "contact.db")
    if os.path.exists(contact_in_storage):
        return db_storage
    if os.path.exists(contact_in_base):
        return base
    return db_storage  # 默认值，即使不存在也以此为准


DEFAULT_DB_DIR = _auto_detect_db_dir()
DEFAULT_OUT_DIR = os.path.join(os.getcwd(), "wechat_export")



def get_runtime_config():
    """读取运行时配置，优先使用环境变量，未设置则回退默认值。"""
    root_base = os.environ.get("WECHATLLM_ROOT_BASE", DEFAULT_ROOT_BASE)
    db_dir = os.environ.get("WECHATLLM_DB_DIR", DEFAULT_DB_DIR)
    out_dir = os.environ.get("WECHATLLM_OUT_DIR", DEFAULT_OUT_DIR)
    return {
        "ROOT_BASE": root_base,
        "DB_DIR": db_dir,
        "MSG_DIR": os.path.join(db_dir, "message"),
        "CONTACT_DB": os.path.join(db_dir, "contact", "contact.db"),
        "OUT_DIR": out_dir,
        "IMG_OUT_DIR": os.path.join(out_dir, "images"),
    }


def refresh_runtime_config(root_base=None, db_dir=None, out_dir=None):
    """按需更新环境变量并刷新模块级路径常量。"""
    if root_base is not None:
        os.environ["WECHATLLM_ROOT_BASE"] = root_base
    if db_dir is not None:
        os.environ["WECHATLLM_DB_DIR"] = db_dir
    if out_dir is not None:
        os.environ["WECHATLLM_OUT_DIR"] = out_dir

    runtime_config = get_runtime_config()

    global ROOT_BASE, DB_DIR, MSG_DIR, CONTACT_DB, OUT_DIR, IMG_OUT_DIR
    ROOT_BASE = runtime_config["ROOT_BASE"]
    DB_DIR = runtime_config["DB_DIR"]
    MSG_DIR = runtime_config["MSG_DIR"]
    CONTACT_DB = runtime_config["CONTACT_DB"]
    OUT_DIR = runtime_config["OUT_DIR"]
    IMG_OUT_DIR = runtime_config["IMG_OUT_DIR"]

    return runtime_config


_runtime_config = refresh_runtime_config()

# MacOS 微信图片缓存格式(dat)的魔法文件头对应其本身的扩展名
MAGIC_MAP = {
    b'\xff\xd8\xff': '.jpg',
    b'\x89\x50\x4e\x47': '.png',
    b'GIF8': '.gif'
}

def decompress_zstd(data):
    """尝试将 zstd 压缩的微信长消息解压为真实文本"""
    if zstd is None:
        return "[提示: 检测到了Zstd压缩的长文本，但未安装zstandard库。请执行 pip3 install zstandard]"
    try:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data).decode('utf-8', errors='ignore')
    except Exception as e:
        return f"[解压彻底失败: {e}]"

def find_group_username(group_keyword):
    """根据群昵称或备注模糊搜寻群ID (后缀 @chatroom)"""
    conn = sqlite3.connect(CONTACT_DB)
    c = conn.cursor()
    c.execute("SELECT username, nick_name, remark FROM contact WHERE (nick_name LIKE ? OR remark LIKE ?) AND username LIKE '%@chatroom'", ('%'+group_keyword+'%', '%'+group_keyword+'%'))
    results = c.fetchall()
    conn.close()
    return results

def load_contacts():
    """搜寻花名册：用于将发言者的 wxid 转换为其真是备注/昵称"""
    conn = sqlite3.connect(CONTACT_DB)
    c = conn.cursor()
    c.execute("SELECT username, nick_name, remark FROM contact")
    mapping = {}
    for row in c:
        username, nick, remark = row
        display = remark if remark else nick
        mapping[username] = display if display else username
    conn.close()
    return mapping

def find_all_table_dbs(table_name):
    """自动扫描所有 message_*.db 分片，返回包含目标表的所有库文件列表。
    同一群聊的历史消息可能分布在多个分片中，必须全部合并才能导出完整记录。"""
    db_files = glob.glob(os.path.join(MSG_DIR, "message_*.db"))
    if not db_files:
        logger.warning("未发现消息分片数据库: %s", MSG_DIR)
        return []

    def sort_key(path):
        base = os.path.basename(path)
        m = re.match(r"message_(\d+)\.db$", base)
        if m:
            return (0, int(m.group(1)))
        return (1, base)

    matched = []
    for db_file in sorted(db_files, key=sort_key):
        try:
            with sqlite3.connect(db_file) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                if c.fetchone()[0] > 0:
                    matched.append(db_file)
        except sqlite3.Error as err:
            logger.warning("扫描消息数据库失败: %s (%s)", os.path.basename(db_file), err)
    return matched


# 兄弟函数，保留下来内部兼容
def find_table_db(table_name):
    results = find_all_table_dbs(table_name)
    return results[0] if results else None

def find_image_file(md5_id, hash_str):
    """在系统的原始缓存路径递归寻找带给定 hash 的所有源图片文件"""
    attach_dir = os.path.join(ROOT_BASE, "msg", "attach", md5_id)
    if not os.path.exists(attach_dir):
        return None
    for root, dirs, files in os.walk(attach_dir):
        for f in files:
            # 排除带 .thumb 后缀的极小缩略图
            if hash_str in f and not f.endswith(".thumb"):
                return os.path.join(root, f)
    return None

def export_by_username(username, display_group_name, progress_callback=None):
    """
    核心提取函数，解耦后供命令行和 Web UI 共同使用
    """
    safe_group_name = re.sub(r'[\\/:*?"<>|]', '_', display_group_name)
    
    # 获取该群聊所在的数据表名称 (Msg_ 加上群ID的 MD5)
    md5_hash = hashlib.md5(username.encode()).hexdigest()
    table_name = f"Msg_{md5_hash}"
    
    db_file = find_table_db(table_name)  # 先搞清楚是否根本有记录
    if not db_file:
        return False, f"查无数据库表，聊天记录可能已被清理或未生成: {table_name}"

    # 收集全部包含该群聊的分片（可能有多个）
    all_dbs = find_all_table_dbs(table_name)
    logger.info("群聊 %s 分布在 %d 个分片: %s",
                display_group_name, len(all_dbs),
                [os.path.basename(d) for d in all_dbs])

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    out_md_file = os.path.join(OUT_DIR, f"聊天记录_{safe_group_name}.md")

    contacts = load_contacts()

    # 建立发送方 ID 映射（从第一个分片读取，各分片共用）
    id2name = {}
    try:
        with sqlite3.connect(db_file) as conn:
            for rowid, uname in conn.execute("SELECT rowid, user_name FROM Name2Id"):
                id2name[rowid] = uname
    except Exception as err:
        logger.warning("读取 Name2Id 映射失败，发信人名称可能不完整: %s", err)

    # 收集所有分片的消息，合并后按时间排序
    messages = []
    for shard_db in all_dbs:
        try:
            with sqlite3.connect(shard_db) as conn:
                c = conn.cursor()
                # 各分片也读取 Name2Id（不同分片可能有各自的映射）
                try:
                    for rowid, uname in c.execute("SELECT rowid, user_name FROM Name2Id"):
                        id2name.setdefault(rowid, uname)
                except Exception:
                    pass
                c.execute(
                    f"SELECT create_time, local_type, real_sender_id, message_content, "
                    f"packed_info_data FROM {table_name} ORDER BY create_time ASC"
                )
                messages.extend(c.fetchall())
        except Exception as e:
            logger.warning("读取分片 %s 失败: %s", os.path.basename(shard_db), e)

    # 全局按时间排序（不同分片可能时间交叉）
    messages.sort(key=lambda r: r[0])

    total_messages = len(messages)
    if progress_callback:
        progress_callback(0, total_messages, f"载入 {display_group_name} 的 {total_messages} 条记录...")
    
    with open(out_md_file, 'w', encoding='utf-8') as f:
        f.write(f"# 聊天记录: {display_group_name}\n\n")
        f.write(f"> 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for idx, msg in enumerate(messages):
            if progress_callback and idx % 20 == 0:
                progress_callback(idx, total_messages, f"处理中: {idx}/{total_messages}...")
            
            create_time, msg_type, sender_id, content, packed_info = msg
            dt = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')
            
            real_username = id2name.get(sender_id, None)
            md_content = ""
            
            if msg_type == 1:
                if type(content) is bytes and content.startswith(b'\x28\xb5\x2f\xfd'):
                    md_content = decompress_zstd(content)
                elif type(packed_info) is bytes and packed_info.startswith(b'\x28\xb5\x2f\xfd'):
                    md_content = decompress_zstd(packed_info)
                else:
                    try:
                        md_content = content.decode('utf-8') if type(content) is bytes else str(content)
                    except Exception:
                        md_content = str(content)
                if md_content == "None": md_content = ""
                
            elif msg_type == 3:
                img_path_md = "[图片未能提取]"
                if packed_info:
                    try:
                        hashes = re.findall(rb'[a-f0-9]{32}', packed_info)
                        if hashes:
                            img_hash = hashes[0].decode('ascii')
                            src_img = find_image_file(md5_hash, img_hash)
                            if src_img:
                                with open(src_img, 'rb') as imgf:
                                    img_data = imgf.read()
                                
                                # 检查是否为较新的 macOS 微信 V2 端对端AES加密。文件头 07 08 56 32
                                if img_data.startswith(b'\x07\x08V2'):
                                    img_path_md = "📸 `[加密高级图片: 微信 V2 协议封锁，请前往电脑端原生查看]`"
                                else:
                                    # V1 单字节异或加密 / 原生未加密图片处理
                                    out_ext = ".jpg" # 默认 fallback
                                    out_data = bytearray(img_data)
                                    
                                    # 探测是纯图还是由单字节异或 (XOR) 加密
                                    found_key = -1
                                    
                                    # 1. 直接无加密明文情况
                                    for head, ext in MAGIC_MAP.items():
                                        if img_data.startswith(head):
                                            found_key = 0
                                            out_ext = ext
                                            break
                                            
                                    # 2. 属于 XOR 图片的判定情况：(data[0] ^ magic[0]) == (data[1] ^ magic[1])
                                    if found_key == -1:
                                        for head, ext in MAGIC_MAP.items():
                                            if (img_data[0] ^ head[0]) == (img_data[1] ^ head[1]):
                                                found_key = img_data[0] ^ head[0]
                                                out_ext = ext
                                                break
                                    
                                    # 解密写入
                                    dest_path = os.path.join(IMG_OUT_DIR, f"{img_hash}{out_ext}")
                                    if found_key != -1 and found_key != 0:
                                        for i in range(len(img_data)):
                                            out_data[i] ^= found_key
                                            
                                    with open(dest_path, 'wb') as dest_f:
                                        dest_f.write(out_data)
                                        
                                    img_path_md = f"![图片](images/{img_hash}{out_ext})"
                            else:
                                img_path_md = f"[缺失本地图片缓存文件: {img_hash}]"
                    except Exception as e:
                        logger.warning(
                            "解析图片消息失败，消息将以占位文本导出: %s",
                            type(e).__name__,
                        )
                        img_path_md = f"[图片提取失败: {type(e).__name__}]"
                md_content = img_path_md
                
            # 语音格式
            elif msg_type == 34: md_content = "🔉 `[语音]`"
            
            # 视频格式
            elif msg_type == 43: md_content = "🎬 `[视频]`"
            
            # 表情包类型
            elif msg_type == 47: md_content = "😀 `[表情包/动画表情]`"
            
            elif msg_type == 49:
                md_content = "📎 `[文件/链接/小程序]`"
                raw = content
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode('utf-8', errors='ignore')
                    except Exception:
                        raw = ''
                if raw and isinstance(raw, str):
                    # 提取 title
                    m = re.search(r'<title>([^<]+)</title>', raw)
                    if m:
                        md_content += f" - **{m.group(1).strip()}**"
                    # 引用/回复消息：提取被引用文本
                    q = re.search(r'<content>([^<]{1,200})</content>', raw)
                    if q:
                        md_content += f"\n> 引用: {q.group(1).strip()}"
                    
            # 内部服务提示 (包括撤回、入群出群提醒等)
            elif msg_type == 10000:
                # 也有可能是 zstd 压缩提醒
                if type(content) is bytes and content.startswith(b'\x28\xb5\x2f\xfd'):
                    md_content = f"<i>{decompress_zstd(content)}</i>"
                elif type(packed_info) is bytes and packed_info.startswith(b'\x28\xb5\x2f\xfd'):
                    md_content = f"<i>{decompress_zstd(packed_info)}</i>"
                else:
                    try:
                        c_str = content.decode('utf-8') if type(content) is bytes else str(content)
                    except Exception:
                        c_str = str(content)
                    md_content = f"<i>{c_str}</i>"
                    
            # 其他极边缘的消息流
            else: md_content = f"📦 `[未知格式内建消息: 类型代码 {msg_type}]`"
            
            # 后续提取出真正文本后，因为 Mac 微信群聊的正文前总是跟着发信人 wxid:\n 的格式，需在此剥离
            if (msg_type == 1 or msg_type == 49) and type(md_content) is str and ':\n' in md_content:
                parts = md_content.split(':\n', 1)
                if len(parts) == 2 and not ' ' in parts[0] and len(parts[0]) < 40 and ('wxid_' in parts[0] or parts[0].endswith('@chatroom')):
                    real_username = parts[0]
                    md_content = parts[1]
                    
            if not real_username:
                real_username = "系统提示" if msg_type == 10000 else "我 (自己)"
                
            sender_display = contacts.get(real_username, real_username)
                
            md_content = md_content.replace('\n', '<br>')
            f.write(f"**{sender_display}** `[{dt}]`\n{md_content}\n\n")

    if progress_callback:
        progress_callback(total_messages, total_messages, "提取完毕！")

    return True, out_md_file

def cli_main(group_keyword):
    groups = find_group_username(group_keyword)
    if not groups:
        print(f"❌ 未找到名称包含 '{group_keyword}' 的群聊。")
        return
    
    if len(groups) > 1:
        print(f"⚠️ 找到多个群聊，默认匹配并导出第一个: {groups[0][1]}")
    else:
        print(f"✅ 找到精准群聊匹配: {groups[0][1]}")
        
    username, nick_name, remark = groups[0]
    display_group_name = remark if remark else nick_name
    
    print(f"开始离线提取: {display_group_name} ...")
    success, msg = export_by_username(username, display_group_name)
    if success:
         print(f"🎉 导出完成！你想要找的聊天记录已完美保存为 Markdown。查看: {msg}")
    else:
         print(f"❌ 导出失败: {msg}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 export_group.py <群聊名称关键字或者名称>")
        print("例如：python3 export_group.py '公共技术部'")
        sys.exit(1)
    cli_main(sys.argv[1])
