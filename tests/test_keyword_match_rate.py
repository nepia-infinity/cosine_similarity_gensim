import spacy
from typing import Set
from  test_extract_normalized_keywords import extract_normalized_keywords

# GiNZAモデルのロード
try:
    # 実行環境に合わせて 'ja_ginza' または 'ja_ginza_electra' を使用
    nlp = spacy.load("ja_ginza_electra")
except OSError as e:
    print(f"エラー: GiNZAモデルが見つかりません。事前にインストールしてください。詳細: {e}")
    print("pip install spacy ja-ginza-electra")
    nlp = None

# 形態素解析の対象とする品詞
TARGET_POS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']

def tokenize(text: str) -> Set[str]:
    """
    GiNZAを使用してテキストを形態素解析し、重要な単語のセットを返す。
    単語はすべて小文字に変換される。
    """
    if nlp is None:
        return set()
    
    doc = nlp(text)
    tokens = {
        token.lemma_.lower()  # 見出し語を小文字に変換
        for token in doc
        if token.pos_ in TARGET_POS and not token.is_stop
    }
    return tokens

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    2つのテキスト間のJaccard係数（キーワード一致率）を計算する。
    Jaccard係数 = |A ∩ B| / |A ∪ B|
    """
    tokens1 = extract_normalized_keywords(text1)
    tokens2 = extract_normalized_keywords(text2)

    if not tokens1 and not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    # ゼロ除算を回避
    if not union:
        return 0.0
        
    similarity = len(intersection) / len(union)
    
    print(f"Text 1: '{text1}'")
    print(f"Tokens 1: {tokens1}")
    print(f"Text 2: '{text2}'")
    print(f"Tokens 2: {tokens2}")
    print("=" * 24)
    print(f"共通キーワード: {intersection}")
    print(f"全キーワード: {union}")
    print(f"Jaccard係数(2つの「集合」Tokensがどれだけ似ているかを計測する指標): {similarity:.4f}\n") 
    
    return similarity

if __name__ == "__main__":
    if nlp is None:
        exit()

    print("--- キーワード一致率の計算テスト ---\n")

    # --- テストケース1: 類似した内容 ---
    text_a = "パソコンが起動しない"
    text_b = "PCが立ち上がらない"
    calculate_jaccard_similarity(text_a, text_b)

    # --- テストケース2: 一部が共通 ---
    text_c = "ノートパソコンの画面が映らない"
    calculate_jaccard_similarity(text_a, text_c)

    # --- テストケース3: 全く異なる内容 ---
    text_d = "スマートフォンのバッテリー交換"
    calculate_jaccard_similarity(text_a, text_d)
    
    # --- テストケース4: 語順が違うが内容は同じ ---
    text_e = "起動しない、私のパソコン"
    calculate_jaccard_similarity(text_a, text_e)

    # --- テストケース5: 大文字と小文字の比較 ---
    text_f = "Windowsのアップデートに失敗した"
    text_g = "windowsの更新ができない"
    calculate_jaccard_similarity(text_f, text_g)
