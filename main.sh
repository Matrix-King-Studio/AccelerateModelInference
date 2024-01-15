# 获取当前路径并保存到变量中
current_path=$(pwd)

# 判断/tmp目录下是否已经存在 Qwen-7B-gguf 文件夹，如果不存在，则执行git clone下载模型
if [ ! -d "/tmp/Qwen-7B-gguf/" ]; then
  echo "下载 Qwen-7B-gguf 模型"
  cd /tmp/ && git clone https://huggingface.co/MatrixStudio/Qwen-7B-gguf
  cd "$current_path/llama.cpp" && ./main -m /tmp/Qwen-7B-gguf/qwen7b-q4_0.gguf -p "Human: 请给我讲一个笑话。Assistant:" -n 32 --temp 1
fi

cd "$current_path"
