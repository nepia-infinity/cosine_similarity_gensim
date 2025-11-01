# test_confirm_current_model.py
from gensim.models import Doc2Vec
import os

def test_confirm_current_model(model_path: str):
    """
    指定されたDoc2Vecモデルの情報を表示する確認用関数。
    モデルの基本情報・語彙・代表単語などを出力する。
    """
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return

    model = Doc2Vec.load(model_path)
    print(f"✅ モデルをロードしました: {model_path}\n")

    # 基本情報
    print("📘 モデル情報")
    print(f"  語彙数: {len(model.wv)}")
    print(f"  ベクトル次元数: {model.vector_size}")
    print(f"  学習済み文書数: {len(model.dv)}\n")

    # 語彙サンプル
    print("📖 登録されている単語（上位10件）:")
    for w in list(model.wv.key_to_index.keys())[:10]:
        print(f"  - {w}")
    print()

    # タグサンプル
    print("📄 登録されている文書タグ（上位10件）:")
    print(model.dv.index_to_key[:10])
    print()

    # モデル統計情報
    print("🧠 モデル概要:")
    print(model)
    print("\n✅ モデル確認が完了しました。")


if __name__ == "__main__":
    # 最新のモデルファイルを指定
    model_path = r"C:\Users\nepia\Desktop\gensim\models\doc2vec_model_2025_1031_1949.model"
    test_confirm_current_model(model_path)
