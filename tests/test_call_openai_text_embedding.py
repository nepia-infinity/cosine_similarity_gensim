import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ベクトル化するためのテキストデータを定義
texts = [
    "Mコパの申請方法を教えてください。", #0
    "Microsoft Copilotの申請方法を教えてください。", #1
    "Mコパ（Microsoft Copilot）の申請方法を教えてください。", #2
    "Gemini使いたいです。", #3
    "東宝シネマズ新宿の映画館の没入体験マジでヤバい。",  #4
    "PCが立ち上がらない。", #5
    "Macの電源が付かない。", #6
]

# それぞれのテキストをベクトル化
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeds = [d.embedding for d in response.data]

# コサイン類似度を計算することで、テキストが完全一致していなくても似ているかを推測できる
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Mコパの申請方法 vs Microsoft Copilotの申請方法：", cosine_similarity(embeds[0], embeds[1]))
print("Mコパの申請方法 vs Mコパ（Microsoft Copilot）の申請方法：", cosine_similarity(embeds[0], embeds[2]))
print("Mコパの申請方法 vs Geminiを使いたい：", cosine_similarity(embeds[0], embeds[3]))
print("Mコパの申請方法 vs 東宝シネマズ新宿の映画館の没入体験：", cosine_similarity(embeds[0], embeds[4]))
print("PCが立ち上がらない vs Macの電源が付かない　:", cosine_similarity(embeds[5], embeds[6]))
