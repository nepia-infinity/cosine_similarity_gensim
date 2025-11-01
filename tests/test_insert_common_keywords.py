import os
import pandas as pd
from openai import OpenAI
from typing import Set
from test_extract_normalized_keywords import extract_normalized_keywords



try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("環境変数 'OPENAI_API_KEY' が設定されていません。")
except Exception as e:
    print(f"APIキーの読み込みエラー: {e}")
    print("スクリプトを実行する前に、ターミナルでAPIキーを設定してください。")
    client = None 

# -------------------------------------------------------------------
# 3. メイン処理：CSVを読み込み、キーワードを書き込む
# -------------------------------------------------------------------
def preprocess_csv(input_file: str, output_file: str, text_column: str):
    """
    CSVを読み込み、指定された列のテキストを正規化し、
    結果を新しい列に保存する。
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"エラー: {input_file} の読み込みに失敗しました。 {e}")
        return

    print(f"CSV ({input_file}) の読み込み成功。{len(df)}件の処理を開始します...")
    
    processed_keywords = []
    total = len(df)
    
    for i, text in enumerate(df[text_column]):
        print(f"処理中... {i+1} / {total} 件目")
        
        # インポートした関数を使用
        keywords_set = extract_normalized_keywords(text)
        
        # CSV保存のため、Setをカンマ区切りの文字列に変換
        keywords_str = ",".join(keywords_set)
        
        processed_keywords.append(keywords_str)

    
    # 新しい列として 'normalized_keywords' をDataFrameに追加
    df['normalized_keywords'] = processed_keywords
    
    try:
        # 結果を新しいCSVファイルに保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("\n" + "="*30)
        print(f"✅ 処理が完了しました！")
        print(f"新しいファイル: {output_file} を保存しました。")
        print("--- 保存後のデータ (先頭5行) ---")
        print(df.head())
        print("="*30)
    except Exception as e:
        print(f"エラー: {output_file} への書き込みに失敗しました。 {e}")

# -------------------------------------------------------------------
# 4. スクリプトの実行
# -------------------------------------------------------------------
if __name__ == "__main__":
    if client is None:
        print("OpenAI APIキーが設定されていないため、実行を中止します。")
    else:
        # --- ここを修正 ---
        # input_file と output_file に同じパスを指定すると
        # 元のファイルに上書き保存されます。
        
        target_file_path = 'csv/in_house_inquiries_with_embeddings.csv'
        
        preprocess_csv(
            input_file=target_file_path,
            output_file=target_file_path, # 読み込み元と同じファイルパスを指定
            text_column='text'
        )