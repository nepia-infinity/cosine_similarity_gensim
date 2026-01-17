import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import ast

# 環境変数を読み込む
load_dotenv()

# OpenAIクライアントを初期化
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# CSVファイルを読み込む
csv_path = 'csv/in_house_inquiries_with_embeddings.csv'

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"エラー: {csv_path} が見つかりません。")
    print("まず `create_embeddings.py` を実行して、Embeddingを含んだCSVファイルを生成してください。")
    exit()

# --- ここに問い合わせ内容を入力してください ---
input_text = "ノートPCが起動しない"
# -----------------------------------------

print(f"入力されたテキスト: '{input_text}'")

# コサイン類似度を計算する関数
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. 新しい問い合わせ内容のEmbeddingを生成
print("入力されたテキストのEmbeddingを生成中...")
try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[input_text]
    )
    input_embedding = response.data[0].embedding
except Exception as e:
    print(f"OpenAI APIの呼び出し中にエラーが発生しました: {e}")
    exit()

print("Embeddingの生成が完了しました。")

# 2. CSVから既存のEmbeddingを読み込み、文字列からリストに変換
#    CSVに保存されたembeddingは文字列として読み込まれるため、ast.literal_evalでPythonのリストオブジェクトに変換します。
try:
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
except ValueError as e:
    print(f"CSVファイルの'embedding'カラムの変換中にエラーが発生しました: {e}")
    print("CSVファイルの内容が正しいか確認してください。")
    exit()


# 3. 新しいテキストとCSV内の全テキストとの類似度を計算
print("類似度の計算中...")
df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity(input_embedding, emb))

# 4. 類似度が高い順にソートし、上位3件を取得
top_3_similar = df.sort_values(by='similarity', ascending=False).head(3)

# 結果を表示
print("\n--- 類似度の高い問い合わせ トップ3 ---")
for index, row in top_3_similar.iterrows():
    print(f"【類似度: {row['similarity']:.4f}】")
    print(f"  {row['text']}")
    print("-" * 20)

