# 実装方針 (STRATEGY)

仕様書「LLM意味構造解析ツール」の実装戦略を定めるドキュメント。
仕様書 §8「意図的に未確定とした事項」に対する選定理由を含む。

> **更新履歴の概要**: 初期実装は本書の通り `argparse` ベースを想定していたが、
> 実装段階で **typer + rich**（サブコマンド型）に切り替えた。
> 同様に CLI 体系・キャッシュ実装・ディレクトリ構成は実装と差分があり、
> 該当節（§3.4 / §3.5 / §4）はあくまで初期方針の記録として残している。
> 現状コードに対する正式な記述は `README.md` と `LLM意味距離測定アプリ仕様書.md` を参照すること。
> 主な実装上の変更点:
> - CLI: `argparse` → **typer + rich**、サブコマンドは `rank` / `pairs` / **`explore`** の 3 つ
> - キャッシュ: SQLite → `manifest.json` + `vectors.npy`（`rank`/`pairs`）と全層キャッシュ（`explore`）
> - パッケージ管理: 素の `pip` → **`uv`**（`pyproject.toml` + `uv.lock`）
> - anisotropy 対策: `--center` を既定 on（`docs/ANISOTROPY.md`）

## 1. 技術スタック

| 項目 | 採用 | 理由 |
|------|------|------|
| 言語 | Python 3.9+ | LLMエコシステム(transformers, torch)が最も成熟 |
| LLM ランタイム | HuggingFace transformers + torch | 隠れ状態(全層)取得APIが提供されている。`output_hidden_states=True`で各層の hidden state を取り出せる |
| アクセラレータ | Apple Silicon MPS | M1 Macで動作。`torch.device("mps")` |
| 候補絞り込み(最遠探索) | Pagh 2016 Approximate Furthest Neighbor (arXiv:1611.07303) — ランダム射影 + per-point AFN | 理論保証あり ((c, δ)-近似)、PCA 候補プールより全モデルで高精度 (`docs/REPORT_AFN.md`) |
| キャッシュ | sqlite + ベクトル本体は npy/safetensors | 単語×モデルの細粒度キャッシュ。テキストキーと数値ベクトルの両方を扱える |
| CLI | argparse(標準ライブラリ) | サブコマンド方式に十分。外部依存を減らす |
| HTML出力 | Jinja2 風 文字列テンプレート(または f-string)。単一HTML、CSS/JS埋め込み | 単一ファイルで完結、ブラウザ閲覧のみ |
| 図示 | 純粋なHTML/CSS バー + Chart.js (CDN無しでインライン埋込) | サーバー不要、外部通信不要(spec §7) |
| 日本語辞書(語彙ソース) | SudachiPy + SudachiDict-core | サンプル語彙生成用。本番ユーザは任意ファイル入力可 |

## 2. 初期対応モデル

| モデル | 採用 | 理由 |
|--------|------|------|
| **Qwen/Qwen2.5-0.5B** (494M params, 24 layers) | ★第一候補 | 多言語(日本語含む)、軽量(GPU/MPSで動作)、HF直DL可、ローカル全層フックOK |
| sbintuitions/sarashina2.2-0.5b | 控え | 日本語特化だが、初回はQwenで足場を作る |

「全層60%」= 24層 × 0.6 = **第14層**(0-indexedで14)を使用。

## 3. 仕様§8の決定事項

### 3.1 表示指標
- `cos_sim` をそのまま使用（値域 -1..+1、+1 で同義、-1 で正反対）
- 別途の距離指標への変換は行わない（仕様 4.2.1）
- CSV/JSON 出力は `cosine` 列のみ。

### 3.2 上位N探索アルゴリズム (改訂版: Pagh AFN ベース)
- **段階1: 候補絞り込み — Pagh 2016 Approximate Furthest Neighbor**
  - L 個のガウス random unit 方向 (ℓ²-norm 1) で全点を射影
  - 各クエリ点 q について「各方向で q と反対側にある」点を候補に展開 (priority queue + 早期打ち切り)
  - L, m は近似比 c から自動導出: $L = \lceil 2 n^{1/c^2} \rceil$, $m = \lceil 1 + e^2 L \log^{c^2/2 - 1/3} n \rceil$
  - c=1.0 で厳密 (= 総当たり), c=1.5 既定 (高精度), c=2.0 で粗速
- **段階2: 精密比較**
  - 候補ペアそれぞれの cos_sim を計算し、グローバル昇順上位 N を返す
- 5モデルでの実測: c=1.5 で R@2000 ≥ 0.95 を全モデルで達成 (旧PCA手法は ruri/e5 で 0.20〜0.50)
- 語彙数 ≤ EXACT_THRESHOLD (=2000) では常に総当たり (高速・厳密)
- 文献: arXiv:1611.07303 (Pagh et al., ICS 2016); Williams 2018 (SODA, arXiv:1709.05282) — 厳密最遠ペアは高次元 ℓ₂ で O(n²) 未満不可

### 3.3 入力フォーマット
- 語彙ファイル: **1行1単語のUTF-8プレーンテキスト** (空行・#コメント許容)
- 対義語ペアファイル: **1行に「単語A␣単語B」(空白またはタブ区切り)**

### 3.4 ディレクトリ構成
```
WordAntipode/
├── wordantipode/             # Python パッケージ
│   ├── __init__.py
│   ├── cli.py                # argparse のエントリポイント
│   ├── adapters/             # モデルアダプタ
│   │   ├── base.py           # 抽象アダプタ
│   │   └── hf_causal_lm.py   # HF CausalLM 共通アダプタ
│   ├── core/
│   │   ├── vectorizer.py     # 単語→ベクトル
│   │   ├── cache.py          # ベクトルキャッシュ(SQLite)
│   │   ├── distance.py       # 距離計算・変換式
│   │   └── search.py         # 上位N探索(PCA + 精密)
│   ├── io/
│   │   ├── reader.py         # 語彙/対義語ファイル読込
│   │   └── writer.py         # HTML/CSV/JSON 出力
│   └── templates/
│       └── report.html       # 単一HTMLテンプレート
├── data/
│   ├── sample_vocab.txt      # SudachiDictから生成したサンプル語彙
│   └── sample_antonyms.txt   # 既存対義語ペア(明/暗 等)
├── cache/                    # SQLite ベクトルキャッシュ
├── out/                      # 出力先(HTML/CSV/JSON)
├── docs/
│   ├── PLAN.md, STRATEGY.md, KPI.md, TEST.md
├── pyproject.toml            # `pip install -e .` で `wordantipode` コマンド化
└── README.md
```

### 3.5 CLI コマンド体系

**初期方針（参考、実装と差分あり）**:
```
wordantipode rank --model qwen2.5-0.5b --vocab words.txt --top 20 --out out/
wordantipode antonym --model qwen2.5-0.5b --pairs antonyms.txt --out out/
wordantipode models --list                 # 登録アダプタ一覧
wordantipode cache --info | --clear MODEL
```

**現状の実装** (詳細は `README.md`):
```
word-antipode rank    --input <vocab.txt>    --model <hf_id> --top N [--center/--no-center]
word-antipode pairs   --input <pairs.txt>    --model <hf_id>          [--center/--no-center]
word-antipode explore --words "a,b,c,..."    --model <hf_id> [--reference-vocab FILE] [--max-axes K]
```

`models` / `cache` 操作サブコマンドは未実装（モデルは HuggingFace ID 直指定、
キャッシュは `--cache-dir` 配下のディレクトリを直接見る運用）。

## 4. キャッシュ戦略

- SQLite テーブル: `vectors(model_id, layer, normalize, word, vector BLOB, created_at)`
- 主キー: `(model_id, layer, normalize, word)` → 層・正規化方式が変われば自動で別エントリ
- ベクトルは np.float16 で BLOB 保存(0.5Bモデルでも 1024次元程度 × 1万語 = 20MB前後、float32なら2倍)

## 5. 外部通信タスク(先に消化)

1. `pip3 install --user transformers torch huggingface_hub sudachipy sudachidict-core jinja2 numpy scikit-learn` ★ネットワーク
2. Qwen2.5-0.5B モデルウェイトのダウンロード(HF Hub) ★ネットワーク
3. (動作確認のため)サンプル語彙生成 ※辞書を `pip` で入れる時点でデータも付随

これら以降は完全ローカル作業。
