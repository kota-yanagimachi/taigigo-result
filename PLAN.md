# 実装計画 (PLAN)

> **状況**: タスク #1〜#11（rank/pairs を含む基本機能と README）はすべて完了済み。
> 以後 anisotropy 対策の `--center` 既定 on 化、`explore` サブコマンド追加、参照語彙の同梱、
> Sudachi 抽出語彙の同梱、ベクトル化のチェックポイント / レジューム化を行った。
> 現状コードに対する正式な仕様は `LLM意味距離測定アプリ仕様書.md`、
> 戦略は `docs/STRATEGY.md`、観察ノートは `docs/EXPLORE_FINDINGS.md` を参照。
> 本ファイルは初期実装時点のタスク表として歴史的に残しているもの。

## タスク一覧と依存

| # | タスク | 種別 | 依存 | 状態 |
|---|--------|------|------|------|
| 1 | pipパッケージ群インストール | **ASK** (外部通信) | - | done (uv に移行) |
| 2 | LLMモデルダウンロード (Qwen2.5-0.5B) | **ASK** (外部通信) | 1 | done |
| 3 | 計画文書(PLAN/KPI/TEST)作成 | ALLOW | - | done |
| 4 | パッケージ骨格 (pyproject + CLI) | ALLOW | 1 | done |
| 5 | モデルアダプタ (HF Causal LM) | ALLOW | 4 | done |
| 6 | ベクトル化 + SQLiteキャッシュ | ALLOW | 5 | done (npy + manifest.json で実装) |
| 7 | 距離計算 + 上位N探索 | ALLOW | 6 | done |
| 8 | HTML/CSV/JSON 出力 | ALLOW | 7 | done |
| 9 | サンプル語彙・対義語データ生成 | ALLOW | 1 | done (`examples/` + `scripts/extract_sudachi_vocab.py`) |
| 10 | 統合テスト・動作確認 | ALLOW | 2,8,9 | done |
| 11 | README作成 | ALLOW | 10 | done |
| 12 | anisotropy 対策 (`--center` 既定 on) | ALLOW | 8 | done (`docs/ANISOTROPY.md`) |
| 13 | `explore` サブコマンド (全層 PCA + N-gon レーダー) | ALLOW | 12 | done (`docs/EXPLORE_FINDINGS.md`) |
| 14 | 参照語彙の同梱 + 全層キャッシュ | ALLOW | 13 | done |
| 15 | チェックポイント保存 / レジューム | ALLOW | 6 | done |

## 実行順
ASKタスク (#1, #2) を最初にまとめて消化 → 残りのALLOWタスクを #3→#11 の順で連続実行。

## ALLOWタスク同士の細粒度ステップ

### #4 パッケージ骨格
- `wordantipode/__init__.py`, `cli.py`
- `pyproject.toml` (entry_point: `wordantipode = wordantipode.cli:main`)
- サブコマンド: `rank`, `antonym`, `models`, `cache`

### #5 アダプタ
- `adapters/base.py`: `Adapter` ABC (load, vectorize(word) -> np.ndarray, info: dict)
- `adapters/hf_causal_lm.py`: HuggingFace AutoModel + AutoTokenizer
  - `output_hidden_states=True` で全層取得
  - 60%層 = `int(num_hidden_layers * 0.6)` (0-indexed)
  - 最終トークン位置の hidden state
  - L2 正規化
- `adapters/registry.py`: モデルID → アダプタ・コンストラクタの辞書

### #6 キャッシュ
- SQLite DB: `cache/vectors.db`
- スキーマ: `vectors(model_id TEXT, layer INT, normalize TEXT, word TEXT, vector BLOB, PRIMARY KEY(model_id,layer,normalize,word))`
- ベクトルは float16 BLOB
- `get_or_compute(adapter, words: List[str])` でバッチ処理

### #7 類似度・探索
- `metrics.py`: `cosine_matrix`, `cosine_for_pairs`（コサイン類似度をそのまま使用）
- `search.py`:
  - `find_top_n_distant_pairs(vectors, words, n, candidate_size=50000)`
  - 候補生成: PCA で 3次元、各軸の両極 200 語 × 全組合せ + ランダムK ペア
  - 重複ペア除去
  - 上位N返却

### #8 出力
- HTML テンプレ(Jinja風)で
  - メタ情報ブロック (モデル名/層/語彙ファイル名/件数/日時)
  - ランキング表
  - 各ペアにバー(コサイン類似度を -1..+1 でスケール、原点に基準線)
  - 全ペア分布のヒストグラム/散布図 (SVG または Chart.js のインライン埋込)
- 同名で `.csv` `.json` も出力

### #9 サンプルデータ
- `data/sample_vocab.txt`: 日常名詞 100語程度をハードコード or SudachiDict 周辺から抽出
- `data/sample_antonyms.txt`: 古典的対義語 (愛/憎, 明/暗, 上/下, 内/外, 善/悪, ...)

### #10 テスト
- `wordantipode models --list` で qwen2.5-0.5b が出る
- `wordantipode rank --model qwen2.5-0.5b --vocab data/sample_vocab.txt --top 5 --out out/` → HTML, CSV, JSON生成
- 再実行でキャッシュヒット (推論スキップ・キャッシュメッセージ確認)
- `wordantipode antonym --model qwen2.5-0.5b --pairs data/sample_antonyms.txt --out out/` → 対義語距離出力
- N=5 と N=20 の結果が「最初の5件」一致

### #11 README
- 動作要件、インストール手順、各サブコマンド例、出力ファイル説明
