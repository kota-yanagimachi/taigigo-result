# テスト項目

> 実装段階で CLI 名は `word-antipode`、副次サブコマンドは `pairs` に確定。
> 加えて探索サブコマンド `explore` を追加した。本章はそれに合わせて記述する。
> 自動テストはまだ整備していないため、当面は **手動チェックの観点表** として使う。

## 1. ユニット相当の小確認

### 1.1 類似度値
- `cosine` を生値のまま使用、変換は行わない
- ベクトルが L2 正規化済みなら `cosine` は常に -1..+1 範囲

### 1.2 ベクトル形状
- 任意の1単語をベクトル化 → 1次元 numpy 配列, L2ノルム=1.0(±1e-5)
- 多トークン語(例「東京特許許可局」)も同次元で返る

### 1.3 キャッシュ
- 初回: 全件 INSERT (`manifest.json` + `vectors.npy` を新規作成)
- 2回目: 全件キャッシュヒット (HF推論呼ばれず) → 推論呼び出し回数 0
- `--layer-fraction` を変更: `config_fingerprint` が変わるので別ディレクトリで再計算
- 4096 語ごとにチェックポイントが書き出され、途中 kill しても再実行で続きから走る

### 1.4 中心化 (`--center`)
- 既定 on。`--no-center` で生コサインに切り替え可能
- `pairs` で対義語ペアの cos が `--no-center` だと 0.99 付近に張り付き、
  `--center` 既定では値が広がることを目視確認（詳細は `docs/ANISOTROPY.md`）

## 2. CLI 結合テスト

### 2.1 help
```
word-antipode --help
word-antipode rank --help
word-antipode pairs --help
word-antipode explore --help
```
→ 各サブコマンドの説明が表示される

### 2.2 メイン機能(rank)
```
word-antipode rank \
  --input examples/vocab.ja.small.txt \
  --model cyberagent/open-calm-small \
  --top 5
```
**期待:**
- 進捗バー表示
- `out/<日時>__rank__cyberagent__open-calm-small__vocab.ja.small.{html,csv,json}` 生成
- HTMLには5ペア × {単語A, 単語B, cosine, バー}, メタ情報, 全体分布図

### 2.3 N変更でも上位は不変
```
word-antipode rank ... --top 5    # → A,B,C,D,E
word-antipode rank ... --top 20   # → A,B,C,D,E,F,...,T
```
**期待:** 先頭5件が同順序

### 2.4 副次機能(pairs)
```
word-antipode pairs \
  --input examples/pairs.ja.small.txt \
  --model cyberagent/open-calm-small
```
**期待:**
- 全ペアの cosine 一覧 HTML/CSV/JSON（類似度の降順）

### 2.5 キャッシュヒット確認
2回連続で `rank` を実行
- 1回目: 「ベクトル化: N語を新規計算」
- 2回目: 「全N語はキャッシュ済」
- 2回目の実行時間 < 1回目の半分

### 2.6 単一/多トークン語混在
`mixed.txt` に `愛`, `憎`, `東京特許許可局`, `言語学`, `123` を入れて実行
- エラーなく結果が出る

### 2.7 探索機能 (`explore`)
```
word-antipode explore \
  --words 愛,憎しみ,光,闇,生,死,始まり,終わり,喜び,悲しみ,武蔵高校地学部 \
  --model cyberagent/open-calm-small
```
**期待:**
- 単一 HTML が `out/<日時>__explore__cyberagent__open-calm-small.html` に生成
- 開くと:
  - A. PCA 2D 軌跡: ドロップダウンで PC#1..PC#5 を切替できる（既定 PC#2 × PC#3）。層スライダーで「層 0..現在層」までの軌跡を表示。凡例 / 軌跡へのホバーで強調
  - B. K-gon レーダー: 表示軸数スピナー（既定 16）で K を変更できる。軸頂点ホバーで参照語彙の正/負方向 top5 がツールチップ表示。「全て」ボタンで凡例の単語を一括 ON/OFF
  - レーダーと軌跡の層スライダーが双方向同期

### 2.8 参照語彙の差し替え (`explore`)
```
word-antipode explore --words ... \
  --reference-vocab examples/vocab.ja.1000.txt
```
**期待:**
- 参照語彙キャッシュが `<cache_dir>/<model>/explore_reference/<vocab_hash>/` 配下に作られる
- 同じ参照語彙で再実行すると「参照語彙キャッシュ済を読み込み」となり推論が走らない

## 3. 出力確認(目視)

- `rank` / `pairs` の HTML をブラウザで開いて、
  - 順位、単語、意味類似度（cosine）、バー、メタ情報が読み取れる
  - バーチャートの x 軸が -1..+1 で固定され、原点に基準線が引かれている
  - 全体分布図(ヒストグラム or 散布図)が表示される
- `explore` の HTML は外部 CDN を読まずに 1 ファイルで完結し、JS エラーがコンソールに出ない

## 4. 既知の許容事項

- Qwen2.5-0.5B / open-calm-small 等は日本語意味距離の質はモデル相応（仕様 §2.2）
- 類似度の絶対値の解釈は研究的・実験的。順序関係に意味がある
- `explore` の N-gon レーダーは「次元値の生の分布」を見るための可視化であり、
  軸の意味づけ top5 は参照語彙の最大／最小活性を機械的に拾っただけのヒント情報である
