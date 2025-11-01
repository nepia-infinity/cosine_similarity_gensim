import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

texts = [
    "今日はお絵描き教室egaco@千葉の無料体験に出掛けてきました。",
    "今日はお絵描き教室egaco@新宿の無料体験に出掛けてきました。",
    "お絵描きアプリのClip Studio Paintのブラシって最高だよね",
    "東宝シネマズの映画館の没入体験マジでヤバい。",
    "PCが立ち上がらない。",
    "Macの電源が付かない。",
]

# それぞれのテキストをベクトル化
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeds = [d.embedding for d in response.data]

# コサイン類似度を計算
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("お絵描き教室egaco@千葉 vs お絵描き教室egaco@新宿:", cosine_similarity(embeds[0], embeds[1]))
print("お絵描き教室egaco@千葉 vs Clip Studio Paint:", cosine_similarity(embeds[0], embeds[2]))
print("お絵描き教室egaco vs 東宝シネマズ:", cosine_similarity(embeds[0], embeds[3]))
print("PCが立ち上がらない。 vs Macの電源が付かない。:", cosine_similarity(embeds[4], embeds[5]))
