# WechatLLM - Mac 版微信本地知识库提取工具箱 🚀

本项目旨在帮助 Mac 用户从本地电脑原生的微信缓存库中完美解压、提取并结构化所有的微信群组或个人聊天记录。提取后的 Markdown 文件非常适合直接投喂给 GPT / Claude 大语言模型进行个人知识库微调 (Fine-Tuning) 学习和本地私有模型 RAG 召回。

**支持环境**: macOS 微信 v4.x (新款原生版微信)

---

## 🛠 工作原理解析与完整使用指南

提取并无损解密你本地长达几年的数十GB微信聊天记录包含三个关键步骤：

### 🔑 步骤 1：拦截并获取底层微信 SQLCipher 数据库绝密密码

最新版的 Mac 微信启用了极高强度的本地 SQLite (SQLCipher 4) 数据库加密机制，并在你彻底启动、扫码登录时动态生成解密使用的 64 位 16进制密码 （其实是内部衍生推导的秘钥 PBKDF2 结果）。

> [!WARNING]
> **重要前置条件：必须关闭 macOS 的 SIP (系统完整性保护)**
> 
> 由于苹果的安全机制与微信执行文件的硬性加固，如果在 SIP 开启状态下直接挂载微信内存，终端会提示 `attach failed` 被系统强行阻拦。
> 
> **检查你的 SIP 状态：** 在终端运行 `csrutil status`，如果输出是 `System Integrity Protection status: disabled.` 则代表你已满足条件，可直接进入下一步！
> 
> **如果显示 enabled，则需要暂时关闭它：**
> 1. 完全关机。
> 2. 长按电源键不放（Intel Mac 则按住 `Command + R` 开机），直到屏幕出现“正在载入启动选项”后松开。
> 3. 点击“选项”，输入管理员密码进入恢复模式 (Recovery Mode)。
> 4. 在顶部系统菜单栏找到 `实用工具` -> `终端`。
> 5. 输入 `csrutil disable` 并回车，输入 `y` 以及管理员密码确认。
> 6. 再次输入 `reboot` 重启电脑。
> *（注：导出完成后，你可以用同样的方法进入恢复模式输入 `csrutil enable` 重新开启它以保证最强安全性）*

**获取密码步骤 (纯控制台零依赖)：**


1. **彻底退出你的微信**进程 (关闭后状态栏也不能看到微信图标)。
2. 打开你的系统终端 (Terminal / iTerm2)，直接运行以下指令：
   ```bash
   sudo lldb -n WeChat -w
   ```
   *（输入开机密码授权后台最高调试拦截权限。命令回车后它会“卡住”并显示等待 WeChat 进程唤醒。）*
3. **立刻点击打开微信应用程序**。此时由于 lldb 截获，微信客户端的界面多半还没来得及渲染，“卡”在终端里了，并输出 `Process XXXXX stopped`，出现 `(lldb)` 提示符。
4. 现在给负责密码衍生加密的核心内置函数 `CCKeyDerivationPBKDF` 下断点，在 `(lldb)` 提示符处执行：
   ```text
   br set -n CCKeyDerivationPBKDF
   ```
   看到成功反馈 `Breakpoint 1: where = ...` 后，继续输入 `c` (表示 continue 继续运行)：
   ```text
   c
   ```
5. **在微信上用手机扫码正常登录**。由于你在进行认证，登录瞬间系统一定会调用那个加密函数被再次卡死。
6. 回到终端，它应该又落在了 `(lldb)` 的拦截中。最后，读取存储在核心 `x1` 通用寄存器中的长度为 32 字节的解密私钥：
   ```text
   memory read --size 1 --format x --count 32 $x1
   ```
   **你的密码出现了！** 终端打印出的 `0x?? 0x?? 0x??` 共 32 个两位的 16 进制字符，只要去掉开头的 `0x` 全部拼在一起组成非常长的一串无空格字母，即是**你神圣的数据库解码密匙**！
7. 然后在终端中依次输入 `detach` 然后 `quit` 将微信完全释放，使其照常运行。

*(切记将该拼好的密钥妥善保存，例如存在 `wechat_db_key.txt`)*

---

### 🔓 步骤 2：使用密码无损解压并暴露本地隐藏的全部数据库

有了刚才拦截到的 64位十六进制神器密码，就可以直接强行打开加密的微信本体原始数据库。

你的所有本地数据存放于：
`/Users/你的用户名/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/<乱码>/msg/` 和 `contact/` 里。

我们使用工程内编写的 `chatlog` (由于基于 Go 的超高性能解包) 直接把加密的源头拖出来。

```bash
# 进入 chatlog 驱动子目录
cd chatlog

# 编译或执行已经存好的 chatlog_bin 工具进行脱壳解码
# -k 后的双引号填入你刚才费煞苦心抓出来的 64位密码
# -d 传入你个人微信号的具体加密本地路径
./chatlog_bin decrypt -k "a639...你的密码...08342" -p darwin -v 4 -d "/Users/你的用户名/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/moushayiciyahuo_f02f" -o /tmp/wechat_export_test
```

恭喜！在 `/tmp/wechat_export_test/db_storage/` 下，你会看到完全一览无余的解密后真实 `.db` 文件，你可以随时用任意 SQlite 管理软件浏览明文的 `message_0.db` 乃至 `contact.db`！

---

### 📥 步骤 3：(首选) 使用现代化的 Web 可视化面板进行批量提取

为了避免枯燥的纯命令行敲击，本项目为你提供了一个绝佳的 Web 大盘交互界面（基于 Streamlit）。

**环境准备 (推荐使用虚拟环境)：**
```bash
# 1. 在项目根目录创建纯净的 Python 虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境 (每次运行本工具前需执行)
source venv/bin/activate

# 3. 安装依赖包 (Streamlit UI, Pandas 处理引擎等)
pip install -r requirements.txt
```

**一键启动交互大盘：**
```bash
streamlit run app.py
```
*执行后，浏览器会自动弹出一个包含你所有系统已解码**花名册**的神奇控制面板！*
- 🔍 你可以在左侧栏无极**搜索过滤**对象名。
- ☑️ 在列表中自由**勾选**一个或成百上千个聊天对象。
- 🚀 点击一键提取，附带全动态交互进度条，自动压出 Markdown 私人语料池。

所有图片 (除了 V2端到端高强度防泄密不可逆 AES 封锁图片以外) 均会按照时间戳恢复解码插入。

<br>

*(如果你更偏向纯极客风格的命令行操作，不想要 UI，也依然可以直接调用 Python：)*
```bash
python3 export_group.py "公共技术部"
```

### ⚙️ 可选环境变量配置

如果你的解密目录或微信缓存目录不是默认路径，可以在运行前设置以下环境变量（不设置则保持当前默认行为）：

```bash
# 微信原始缓存根目录（用于回溯图片缓存）
export WECHATLLM_ROOT_BASE="/Users/你的用户名/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/你的目录"

# chatlog 解密输出目录（应包含 db_storage/message 与 db_storage/contact）
export WECHATLLM_DB_DIR="/tmp/wechat_export_test/db_storage"

# Markdown 导出目录（默认是当前工作目录下 wechat_export）
export WECHATLLM_OUT_DIR="/Users/你的用户名/Documents/wechat_export"
```

### 🧯 常见失败排查

1. **无法检测到联系人库 (`contact.db`)**
   - 确认你已经执行过 `chatlog_bin decrypt`。
   - 检查 `WECHATLLM_DB_DIR` 是否指向包含 `contact/contact.db` 的 `db_storage` 目录。

2. **导出时报 `查无数据库表: Msg_<md5>`**
   - 该会话可能没有本地消息、被清理，或当前目录并非该账号对应解密结果。
   - 重新确认 `WECHATLLM_DB_DIR` 后重试。

3. **图片提取失败但文本可导出**
   - 部分图片可能为微信 V2 加密图片，或本地缓存已丢失。
   - 此时 Markdown 会保留占位说明，文本消息仍会正常导出。

尽情享用属于你自己的本地超级语料库进行 AI 对话微调与测试！🤖✨
