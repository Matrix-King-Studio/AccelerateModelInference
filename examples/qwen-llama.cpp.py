from llama_cpp import Llama

llm = Llama(model_path="/tmp/Qwen-7B-gguf/qwen7b-q4_0.gguf", logits_all=True)
output = llm(
      "Human: 请给我讲一个笑话。Assistant: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens
      stop=["Human:", "\n", "[end of text]"], # Stop generating just before the model would generate a new question
      echo=True, # Echo the prompt back in the output
      logprobs=31
)
print(output)
