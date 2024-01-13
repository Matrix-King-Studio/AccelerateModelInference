echo "CPU 相关信息"
lscpu

echo "CPU 内存大小"
free -h | grep Mem | awk '{print $2}'

echo "升级 pip"
python -m pip install --upgrade pip

echo "安装 Python 依赖"
pip install -r requirements.txt

# 获取当前路径并保存到变量中
current_path=$(pwd)

# 判断当前目录下是否有 lm-evaluation-harness 文件夹，如果没有的话，执行git clone
if [ ! -d "lm-evaluation-harness" ]; then
  git clone https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && pip install -e .
fi

cd "$current_path"

# 判断当前目录下是否有 llama.cpp 文件夹，如果没有的话，执行git clone
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
fi

cd "$current_path"

cd /tmp/ && git clone https://huggingface.co/Qwen/Qwen-7B

cd "$current_path"