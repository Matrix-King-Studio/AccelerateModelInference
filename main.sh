MODEL_NAME="Qwen-7B-gguf"
MODEL_FILE="qwen7b-q4_0.gguf"
MODEL_USER="MatrixStudio"
DEFAULT_MODEL_BRANCH="main"
MODEL_URL="https://huggingface.co/${MODEL_USER}/${MODEL_NAME}/resolve/${DEFAULT_MODEL_BRANCH}/${MODEL_FILE}"

# 获取当前路径并保存到变量中
current_path=$(pwd)

# 判断/tmp目录下是否已经存在 Qwen-7B-gguf 文件夹，如果不存在，则执行git clone下载模型
if [ ! -d "/tmp/Qwen-7B-gguf/" ]; then
  echo "下载 Qwen-7B-gguf 模型"
  mkdir /tmp/Qwen-7B-gguf && wget -O /tmp/Qwen-7B-gguf/${MODEL_FILE} ${MODEL_URL}
  cd "$current_path/llama.cpp" && ./main -m /tmp/Qwen-7B-gguf/qwen7b-q4_0.gguf -p "Human: 请给我讲一个笑话。Assistant:" -n 32 --temp 1
fi

cd "$current_path"
