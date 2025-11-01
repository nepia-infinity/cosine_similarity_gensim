import pandas as pd
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List, Dict, Any

# GiNZAモデルのロードは一度だけ行う
try:
    # 実行環境に合わせて 'ja_ginza' または 'ja_ginza_electra' などを使用してください
    nlp = spacy.load("ja_ginza_electra")
except OSError as e:
    print(f"エラー: GiNZAモデルが見つかりません。事前にインストールしてください。詳細: {e}")
    nlp = None

# Doc2Vecの学習に使用する品詞
target_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']

# ----------------------------------------------------------------------
# ## 1. データ準備と形態素解析 (Data Preparation)
# ----------------------------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """CSVファイルをロードし、データフレームを返す"""
    try:
        data = pd.read_csv(file_path)
        print(f"データセットをロードしました。総文書数: {len(data)}")
        return data
    except FileNotFoundError:
        print(f"エラー: '{file_path}'が見つかりません。")
        return pd.DataFrame()

def tokenize_data(data: pd.DataFrame) -> List[TaggedDocument]:
    """GiNZAを使用して文書を形態素解析し、TaggedDocumentリストを生成する"""
    if nlp is None:
        return []
        
    print("形態素解析とTaggedDocumentの作成を開始...")

    def tokenize_ginza(text):
        """指定品詞のトークンのみを抽出"""
        doc = nlp(text)
        tokens = [
            token.text
            for token in doc
            if token.pos_ in target_pos and not token.is_stop
        ]
        return tokens

    tagged_data = [
        TaggedDocument(
            words=tokenize_ginza(row['text']),
            tags=[str(row['row_index'])]
        )
        for _, row in data.iterrows()
    ]
    print("TaggedDocumentの作成が完了しました。")
    return tagged_data

# ----------------------------------------------------------------------
# ## 2. モデル学習 (Model Training)
# ----------------------------------------------------------------------

def train_doc2vec_model(tagged_data: List[TaggedDocument]) -> Doc2Vec:
    """TaggedDocumentデータを使用してDoc2Vecモデルを学習させる"""
    
    model = Doc2Vec(
        vector_size=75,       
        window=5,             
        min_count=1, # 1回でも登場した単語を学習モデルに加える       
        workers=4,            
        epochs=40             
    )

    model.build_vocab(tagged_data)
    print("Doc2Vecモデルの学習を開始します...")
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    print("Doc2Vecモデルの学習が完了しました。")
    
    return model

# ----------------------------------------------------------------------
# ## 3. 類似度計算と結果出力 (Similarity Calculation and Output)
# ----------------------------------------------------------------------

def tokenize_data_single(text: str) -> List[str]:
    """単一のテキストを形態素解析する (ヘルパー関数)"""
    if nlp is None:
        return []
    
    doc = nlp(text)
    tokens = [
        token.text
        for token in doc
        if token.pos_ in target_pos and not token.is_stop
    ]
    return tokens

def calculate_similarity_and_rank(model: Doc2Vec, data: pd.DataFrame, query: str) -> List[Dict[str, Any]]:
    """
    クエリ文と学習済み文書との類似度を計算し、上位5件の結果を配列で返す
    """
    if nlp is None:
        return []

    query_tokens = tokenize_data_single(query)
    query_vector = model.infer_vector(query_tokens, epochs=20)

    results = []
    for _, row in data.iterrows():
        doc_id = str(row['row_index'])
        doc_vector = model.dv.get_vector(doc_id)
        
        similarity = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
        )
        
        results.append({
            'row_index': int(doc_id),
            'document': row['text'],
            'similarity': similarity
        })

    sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return sorted_results[:5]

def print_results(top_5_results_array: List[Dict[str, Any]]):
    """上位5件の結果を整形してコンソールに出力する"""
    print("\n-------------------------------------------")
    print("✅ コサイン類似度による類似文書ランキング (上位5件)")
    print("-------------------------------------------")

    print("{:<10} {:<30} {:>10}".format("Row_Index", "Document", "Similarity"))
    print("-" * 52)
    for result in top_5_results_array:
        display_text = result['document']
        if len(display_text) > 25:
            display_text = display_text[:24] + "..."
            
        print("{:<10} {:<30} {:>10.6f}".format(
            result['row_index'],
            display_text,
            result['similarity']
        ))

    print("\n--- 上位5件の結果配列（リスト） ---")
    print(top_5_results_array)

# ----------------------------------------------------------------------
# ## 4. メイン処理の実行
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if nlp is None:
        exit()
        
    query_text = "プログラミングはとても面白いです。Pythonが好き。"
    print(f"🔑入力されたテキスト: '{query_text}'")
    data_file = r'C:\Users\nepia\Desktop\gensim\csv\test_data.csv'
    
    data_df = load_data(data_file) 
    if data_df.empty:
        exit()
        
    tagged_data_list = tokenize_data(data_df)
    doc2vec_model = train_doc2vec_model(tagged_data_list)
    top_5_ranking = calculate_similarity_and_rank(doc2vec_model, data_df, query_text)
    print_results(top_5_ranking)
