from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import spacy
import pytz
from datetime import datetime
from common import get_current_japanese_timestamp
import os

# 1. GiNZAモデルをロード
try:
    # 必要に応じて "ja_ginza" に変えてください
    nlp = spacy.load("ja_ginza_electra")
except OSError as e:
    print("GiNZAモデルが見つかりません。先にインストールしてください。")
    print(e)
    nlp = None

# Doc2Vecで使いたい品詞
TARGET_POS = ["NOUN", "VERB", "ADJ", "ADV"]
BASE_DIR = r"C:\Users\nepia\Desktop\gensim\models"


def load_data(file_path: str) -> pd.DataFrame:
    """CSVを読み込んでDataFrameを返す"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ CSVを読み込みました: {file_path}  総行数: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"❌ CSVが見つかりません: {file_path}")
        return pd.DataFrame()


def to_tagged_docs(df: pd.DataFrame):
    """DataFrame -> TaggedDocument のリストに変換する"""
    if nlp is None:
        return []

    tagged_docs = []

    for _, row in df.iterrows():
        text = str(row["text"])
        row_id = str(row["row_index"]) if "row_index" in df.columns else str(_)

        doc = nlp(text)
        tokens = [
            token.text
            for token in doc
            if token.pos_ in TARGET_POS and not token.is_stop
        ]

        tagged_docs.append(TaggedDocument(words=tokens, tags=[row_id]))

    print(f"✅ TaggedDocument を {len(tagged_docs)} 件 作成しました。")
    return tagged_docs


def save_and_update_model(tagged_data, base_dir="models") -> Doc2Vec:
    """日本時間タイムスタンプ付きファイル名でモデルを保存"""
    # 保存先ディレクトリを作成
    os.makedirs(BASE_DIR, exist_ok=True)

    # 日本時間のタイムスタンプを生成する
    timestamp = get_current_japanese_timestamp()

    # ファイル名を作成
    model_path = os.path.join(BASE_DIR, f"doc2vec_model_{timestamp}.model")
    print(f"モデルを新規作成します: {model_path}")
    
    model = Doc2Vec(
        vector_size=75,
        window=5,
        min_count=1,
        workers=4,
        epochs=40,
    )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=len(tagged_data), epochs=model.epochs)
    model.save(model_path)

    print(f"💾 モデルを保存しました: {model_path}")
    return model


if __name__ == "__main__":
    # あなたのパス
    data_file = r"C:\Users\nepia\Desktop\gensim\csv\test_data.csv"
    model_path = "doc2vec_model.model"

    if nlp is None:
        # 形態素解析できないと学習できないのでここで終了
        raise SystemExit("GiNZA が読み込めないので終了します。")

    # 1. CSVを読む
    df = load_data(data_file)
    if df.empty:
        raise SystemExit("CSVが空だったので終了します。")

    # 2. TaggedDocumentにする
    tagged_docs = to_tagged_docs(df)

    # 3. モデルをつくって保存（既存があれば更新）
    model = save_and_update_model(tagged_docs, BASE_DIR)
    print("🎉 完了しました。")
