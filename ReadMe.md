## 视频跟踪软件使用说明（windows）
#### 1. 本脚本主要由一个不知名的西班牙小哥编写，最初采用的语言是Python2，我在其基础上将其改成Python3，修复了几个Bug，并添加了一个新功能。
#### 2. 安装说明
   - 下载python3安装包[python3][1]，一般可以选择 Windows x86-64 executable installer这个版本
   - 安装python3，安装过程中建议勾选add python to PATH选项，选择Customize installation可以自定义安装路径，其它选项保持默认即可。
   - 安装运行脚本所需要的第三方库，打开命令提示符(或者PowerShell)，输入以下命令
   ```bash
      pip install numpy scipy matplotlib pillow opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   - 下载脚本 [下载地址][2]
   - 将压缩包解压到纯英文路径下，运行文件夹下的run.bat即可开始使用


[1]: https://www.python.org/downloads/release
[2]: https://github.com/alchemist1234/lab-manager/archive/master.zip
