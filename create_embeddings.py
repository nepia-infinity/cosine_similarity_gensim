import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

# 環境変数を読み込む
load_dotenv()

# OpenAIクライアントを初期化
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# CSVファイルを読み込む
input_csv_path = 'csv/in_house_inquiries.csv'
output_csv_path = 'csv/in_house_inquiries_with_embeddings.csv'

try:
    df = pd.read_csv(input_csv_path)
except FileNotFoundError:
    print(f"エラー: {input_csv_path} が見つかりません。")
    exit()

# 'text' カラムが存在するか確認
if 'text' not in df.columns:
    print("エラー: CSVファイルに 'text' カラムが見つかりません。")
    exit()

# テキストのリストを取得
texts = df['text'].tolist()

print("Embeddingの生成を開始します...")

# テキストをベクトル化
try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeds = [d.embedding for d in response.data]
except Exception as e:
    print(f"OpenAI APIの呼び出し中にエラーが発生しました: {e}")
    exit()

print("Embeddingの生成が完了しました。")

# DataFrameにembeddingを追加
df['embedding'] = embeds

# 結果を新しいCSVファイルに保存
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"Embeddingを含んだCSVファイルを {output_csv_path} に保存しました。")

# サンプルとして、最初の2つのテキストのコサイン類似度を計算して表示
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if len(embeds) >= 2:
    similarity = cosine_similarity(embeds[0], embeds[1])
    print("\n--- サンプル類似度計算 ---")
    print(f"1行目: '{texts[0]}'")
    print(f"2行目: '{texts[1]}'")
    print(f"コサイン類似度: {similarity}")
