# 根据具体情况选择合适的pytorch和cuda版本
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 安装git等必要工具
RUN apt update
RUN apt install -y git

# 设定工作空间，拷贝项目代码
WORKDIR /app


CMD ["bash", "-c", "while true; do sleep 10; done"]
