echo "CPU 相关信息"
lscpu

echo "升级 pip"
python -m pip install --upgrade pip

# 判断当前目录下是否有 lm-evaluation-harness 文件夹，如果没有的话，执行git clone
if [ ! -d "lm-evaluation-harness" ]; then
  git clone https://github.com/EleutherAI/lm-evaluation-harness
fi
# 判断 pip list 中是否有 lm-eval，如果没有的话，执行 pip install -e .
if ! pip list | grep 'lm-eval'; then
  cd lm-evaluation-harness && pip install -e . && cd ..
fi

# 判断当前目录下是否有 optimum-benchmark 文件夹，如果没有的话，执行git clone
if [ ! -d "optimum-benchmark" ]; then
  git clone https://github.com/huggingface/optimum-benchmark.git
fi
# 判断 pip list 中是否有 optimum-benchmark，如果没有的话，执行 pip install -e .
if ! pip list | grep 'optimum-benchmark'; then
  cd optimum-benchmark && pip install -e . && cd ..
fi
