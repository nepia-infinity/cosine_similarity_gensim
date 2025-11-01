import os
from openai import OpenAI
from typing import Set

# -------------------------------------------------------------------
# 1. OpenAIクライアントの初期化
# -------------------------------------------------------------------
# 環境変数 'OPENAI_API_KEY' からキーを読み込みます
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("環境変数 'OPENAI_API_KEY' が設定されていません。")
except Exception as e:
    print(f"APIキーの読み込みエラー: {e}")
    print("スクリプトを実行する前に、ターミナルでAPIキーを設定してください。")
    # このままでは実行できないため、サンプル用にダミーのクライアントにします
    client = None 

# -------------------------------------------------------------------
# 2. OpenAIに投げるプロンプトのテンプレート
# -------------------------------------------------------------------
NORMALIZATION_PROMPT_TEMPLATE = """
以下のユーザー(社員)からの問い合わせテキストを読み、その核心的な問題を示す「正規化されたキーワード」を抽出してください。

# 前提条件
* 核心的な「対象（何が）」と「症状（どうなった）」を示すキーワードを抽出します。
* OS名 (Windows, Mac, iOS, Androidなど)、アプリケーション名 (Excel, Slack, Teamsなど)、ハードウェア名 (プリンタ, マウスなど) は、解決方法を特定する上で最も重要なキーワードです。
* Windows, Macが含まれている場合は、キーワードを格納した上で、さらに正規化されたキーワード群にパソコンを追加してください。
* iOS, Androidが含まれている場合は、キーワードを格納した上で、さらに正規化されたキーワード群にスマホを追加してください。
* アプリケーション名 (Excel, Slack, Teamsなど)が含まれている場合は、キーワードを格納した上で、さらに正規化されたキーワード群にアプリケーションを追加してください。
* サービス名 (ジョブカン、バクラクなど)が含まれている場合は、キーワードを格納した上で、さらに正規化されたキーワード群にSaaSを追加してください。
* サービス名 (ジョブカン)が含まれている場合は、キーワードを格納した上で、さらに正規化されたキーワード群に勤怠と労務を追加してください。
* サービス名 (バクラク)が含まれている場合は、キーワードを格納した上で、さらに正規化されたキーワード群に経費精算を追加してください。
* ハードウェア名 (プリンタ, マウスなど)が含まれている場合は、キーワードを格納した上で、さらに正規化されたキーワード群にハードウェアを追加してください。
* 類義語や表記ゆれは、最も一般的で標準的な日本語の単語に「統一（正規化）」してください。
*「PC」「ノートPC」「ノートパソコン」「Desktop」「ラップトップ」 などコンピュータを指す言葉は全て「パソコン」に統一します。
    * 例: 「立ち上がらない」「電源が入らない」 -> 「起動不可」
    * 例: 「映らない」「表示されない」 -> 「表示不可」
    * 例: 「アップデート」「update」 -> 「更新」
* 抽出したキーワードを、カンマ（,）区切りで出力してください。

# 制約条件
* 問い合わせの「挨拶」や「感情表現」（例：お疲れ様です、よろしくお願いします）などは無視してください。
* 正規化キーワード: という単語や改行\nを含めないでください。

# 問い合わせテキスト
{text_input}

# 正規化キーワード
"""

# -------------------------------------------------------------------
# 3. テキストを正規化・キーワード抽出する関数
# -------------------------------------------------------------------
def extract_normalized_keywords(text: str) -> Set[str]:
    """
    OpenAI APIを使用し、テキストを正規化されたキーワードのセットに変換する。
    """
    if not client:
        print("--- (スキップ) OpenAIクライアントが初期化されていません ---")
        return {"(APIキー未設定のためスキップ)"}

    # 1. 入力テキストをプロンプトに埋め込む
    prompt = NORMALIZATION_PROMPT_TEMPLATE.format(text_input=text)
    
    try:
        # 2. OpenAI API (Chat Completions) を呼び出す
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "あなたは、社内ITサポートの問い合わせ内容を分析するエキスパートです。"},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.0 にすることで、指示に忠実な結果を返します
            temperature=0.0, 
            max_tokens=100
        )
        
        # 3. レスポンスからキーワード文字列を取得
        # 例: "パソコン, 起動不可"
        raw_keywords_str = response.choices[0].message.content.strip()
        
        # 4. 文字列をクリーニングし、Setに変換
        if not raw_keywords_str:
            return set() # 何も返ってこなかった場合

        # カンマで分割し、前後の空白を削除
        keywords_list = [kw.strip() for kw in raw_keywords_str.split(',')]
        
        # 空の文字列（例: "a,,b" のような場合）を除外してセットにする
        keywords_set = {kw for kw in keywords_list if kw} 
        
        return keywords_set

    except Exception as e:
        print(f"--- OpenAI APIエラーが発生しました ---")
        print(f"エラー内容: {e}")
        print(f"対象テキスト: {text}")
        return set() # エラー時は空のセットを返す

# -------------------------------------------------------------------
# 4. 実行サンプル
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    print("--- OpenAI APIによるキーワード正規化テスト ---")
    
    # テストしたいテキストのリスト
    test_texts = [
        "PCが立ち上がらない",
        "会社のノートパソコンの電源が全然入らなくて困ってます。",
        "windowsの更新ができないみたい",
        "スマートフォンのバッテリー交換をお願いしたい"
    ]
    
    for text in test_texts:
        print(f"\n■ Input Text:\n  {text}")
        
        keywords = extract_normalized_keywords(text)
        print(f"✅ Normalized Keywords (Set):\n  {keywords}")