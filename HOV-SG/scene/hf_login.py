import os
from huggingface_hub import login

# token = 请在这里填入你的 Hugging Face Token (hf_xxxx)

try:
    login(token=token)
    print("\n✅ 登录成功！你可以继续运行 SAM3 代码了。")
except Exception as e:
    print(f"\n❌ 登录失败: {e}")
