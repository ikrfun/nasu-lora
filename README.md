# nasu-lora
那須ハッカソン用のレポジトリ。　那須に関する情報や語尾が🍆になっているllmを学習する。

## 目的

小型ローカルモデルに那須らしい会話スタイルを学習させ、LoRA・QLoRA・DoRAの違いを比較する。

Agentを作れることに加えて、モデル適応学習の設計、データセット作成、学習、評価まで扱えることを示す。

## 採用モデル

- ベースモデル：`google/gemma-4-E4B-it`
- 学習環境：RTX 5090搭載GPUサーバー
- 学習基盤：Unslothを第一候補とする
- 実験管理：Weights & Biases（W&B）を第一候補とする

## 比較する手法

- LoRA：通常の低ランク適応
- QLoRA：4bit量子化したベースモデルへのLoRA適応
- DoRA：重みの大きさと方向を分けた適応
- 余力があればQDoRA（量子化ベースモデル＋DoRA）も試す

同じベースモデル、同じデータセット、同じ評価質問を使って比較する。

## データセット方針

- 初回は100件程度で作成する
- 学習用と評価用を分離する
- `messages`形式のJSONLを基本とする
- キャラクター実験ではsystem promptを使わず、user → assistantの回答例から口調を学習させる
- Gemma 4のchat templateは手書きせず、tokenizer/Unsloth側で学習時・推論時に同じものを適用する
- 🍆、語尾「なす」、那須らしい応答を含める
- データセットのJSONL骨格・分類・テンプレートはAgentが作成する
- 最終的な質問・回答の編集、那須情報の事実確認、口調の自然さの調整は本人が行う
- 初回データは自作・人手編集を基本とし、出所とライセンスを記録する
- 蒸留や合成データを使う場合は、教師モデルの規約でモデル学習が許可されていることを確認する

### データセット例

```json
{"messages":[{"role":"user","content":"那須のおすすめを教えて"},{"role":"assistant","content":"那須どうぶつ王国がおすすめなす🍆"}]}
```

## データ・モデルの管理方針

Gitを編集・レビュー・再現性管理の正本とし、Hugging Faceは学習済みモデルと安定版データセットの公開・配布先として使い分ける。

- Git：データセット、学習スクリプト、設定、評価コードを管理する
- 学習PC：Gitのtagまたはcommit hashを固定してデータを取得する
- 実験記録：データセットrevision、Git commit、ベースモデル、手法、thinking設定、system prompt設定を記録する
- Hugging Face：完成したLoRA adapterを優先して公開し、必要に応じてmerged modelやDataset Repositoryも公開する
- 公開前：事実情報、外部文章の転載、ライセンス、教師モデルの利用規約、評価データの公開可否を確認する
- 大容量の画像・音声・モデルファイルが増えるまでは、Git LFSやDVCは導入しない

学習済みモデルとデータセットは、どのGit commit・データセットrevisionから作ったかをModel Cardや実験記録に残す。`main`の最新版を暗黙に使わず、tagまたはcommit hashを指定して再現性を確保する。

## 実験管理

学習率などの条件はスクリプトに直書きせず、設定ファイルで管理する。

記録する項目：

- 手法（LoRA／QLoRA／DoRA）
- 学習率、epoch、batch size、gradient accumulation
- LoRA rank、alpha、dropout
- max sequence length、seed
- データセットバージョンと件数
- train loss、eval loss、VRAM、学習時間
- 評価質問への回答結果

## 作業予定

- [x] データセットの骨格を作成する
- [ ] 本人がデータセットを編集・事実確認する
- [x] UnslothでGemma 4 E4Bを読み込む
- [x] 小さなデータでQLoRAを完走させる
- [x] 評価用スクリプトを作成する
- [ ] LoRA・QLoRA・DoRAを同一条件で比較する
- [ ] 実験結果をW&BとREADMEに記録する
