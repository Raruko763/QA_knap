# QA_knap

量子アニーリング (Fixstars Amplify) を用いて車両経路問題 (VRP) を解く実験用プロジェクトです。Julia スクリプトで VRP インスタンスをクラスタ分割し、Python で各クラスタおよび重心間の巡回セールスマン問題 (TSP) を解きます。付属スクリプトで複数インスタンスの一括実行や自動リトライも行えます。

## ディレクトリ構成

```
data/                # サンプル VRP インスタンスとクラスタ結果
  before_data.json   # Julia で生成したクラスタリング結果の例
  raw/               # VRPLIB 形式のインスタンスファイル置き場
out/                 # 実行結果の保存先 (run_division.rb が日付別に作成)
scripts/             # 実行補助スクリプト (Ruby / zsh)
src/                 # Julia / Python 実装
  knapsack.jl        # VRP をクラスタに分割 (Knapsack 近似)
  Qknapcore.py       # Amplify を使ったクラスタ調整＆TSP 解法
  knap_divpro.py     # クラスタ間の入れ替えを行う準備問題
  TSP.py             # Amplify ベースの TSP ソルバ
  vrpfactory.py      # VRP 関連ユーティリティ
```

## 必要な環境

- Julia 1.10 以上 (PyCall, JSON, DotEnv, Statistics, Plots などを利用)
- Python 3.10 以上
  - amplify
  - numpy
  - matplotlib
  - pandas
  - vrplib
- Ruby 3.x (オプション: `scripts/run_division.rb` で使用)
- zsh (オプション: `scripts/run_qknap.zsh` で使用)
- Fixstars Amplify の API トークン

> **メモ:** `src/Qknapcore.py` 内の `self.client.token` にはダミー値が入っています。自分のトークンに置き換えるか、環境変数から読み込むように修正してください。

## セットアップ

1. Python ライブラリをインストールします。
   ```bash
   pip install amplify numpy matplotlib pandas vrplib
   ```
2. Julia パッケージを準備します。初回起動時に以下を REPL で実行しておくと依存が揃います。
   ```julia
   using Pkg
   Pkg.add(["PyCall", "JSON", "DotEnv", "Statistics", "Plots"])
   ```
3. `.env` に出力パスなどを設定するとスクリプト実行が楽になります (任意)。
   ```dotenv
   SAVE_PATH=./out
   INSTANCE_PATH=/path/to/vrplib/files
   AMPLIFY_TOKEN=xxxxxxxxxxxxxxxx
   ```

## 使い方

### 1. VRP インスタンスのクラスタ分割 (Julia)

`src/knapsack.jl` は VRPLIB 形式のインスタンスを読み込み、Knapsack 由来のヒューリスティックで需要量が容量を超えないようにクラスタを構築します。  
出力は各クラスタの都市番号・座標・需要・重心などを含む JSON です。

```bash
julia src/knapsack.jl <保存ディレクトリ> <インスタンスパス>

# 例:
julia src/knapsack.jl out/20240902 QA/data/raw/E-n51-k5.vrp
```

- `保存ディレクトリ` に `before_data.json` が書き出されます。
- スクリプト内の `instance_dir` はデフォルトで固定パスになっているので、`.env` の `INSTANCE_PATH` を読み込むように書き換えるか、直接パスを修正してください。

### 2. クラスタを用いた QA ソルブ (Python)

Julia が生成した `before_data.json` を入力に Amplify を用いてクラスタの改善と各クラスタの TSP を解きます。

```bash
python3 src/Qknapcore.py -j <クラスタJSON> -sp <出力ディレクトリ> --t 2000 -nt 5
```

- `-j` : `knapsack.jl` の出力 JSON パス
- `-sp`: 結果保存先 (クラスタごとのルートやログを JSON 出力)
- `--t`: Amplify のタイムアウト (ミリ秒)
- `-nt`: サンプル数

### 3. バッチ実行スクリプト (任意)

- `scripts/run_division.rb`  
  `.env` の `SAVE_PATH` を基準に、複数の VRP インスタンスを順次 `julia` に投げて `before_data.json` を作ります。
  ```bash
  ruby scripts/run_division.rb
  ```

- `scripts/run_qknap.zsh`  
  `before_data.json` を 10 回まで自動リトライしながら `Qknapcore.py` に渡します。`-j` 引数などはスクリプト内で固定なので、用途に合わせて編集してください。

## 入出力データ

- `before_data.json` 形式 (抜粋)
  ```json
  {
    "cluster_0": {
      "cities": [0, 12, 47, ...],
      "demand": [0, 14, 23, ...],
      "total_demand": 112,
      "capacity": 130,
      "gravity": { "x": 734.8, "y": 421.2 },
      "cluster_distance": [[0, 34, ...], ...],
      "coordinates": [
        { "x": 300, "y": 500 },  // depot
        { "x": 734, "y": 412 },
        ...
      ],
      "required_trucks": 1
    }
  }
  ```
- `Qknapcore.py` の出力  
  `TSP_cluster_results_<日時>.json` に重心 TSP、クラスタ調整 (knap_dippro)、各クラスタ TSP の指標が格納されます。

## トラブルシューティングメモ

- Julia から Python (PyCall) を使うため、Python 側に必要なモジュールがインストールされていることを確認してください。必要なら `ENV["PYTHON"]` を設定してから `Pkg.build("PyCall")` を実行します。
- Fixstars Amplify のトークンが正しく設定されていない場合、`Qknapcore.py` 実行時に認証エラーになります。
- VRPLIB インスタンスは `data/raw` のようなパスで参照できます。手元に大量のファイルがある場合は `.env` の `INSTANCE_PATH` を使うと便利です。

## ライセンス

未定。利用ポリシーが決まっていない場合は、このリポジトリを公開する前にプロジェクトオーナーに確認してください。

