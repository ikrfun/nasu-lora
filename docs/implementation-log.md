# 実装ログ

## 2026-07-27 — UnslothによるGemma 4 E4B LoRA準備状況

- プロジェクトを確認した。`data/train.jsonl` は80件、`data/eval.jsonl` は20件で、いずれも `messages` 形式の `user → assistant` 会話。
- READMEの採用モデルは `google/gemma-4-E4B-it`、学習基盤はUnsloth第一候補。
- ikrfun-wsを読み取り専用で確認した。RTX 5090（32GB）とRTX 4090（24GB）はほぼ空き状態。
- ikrfun-wsの既存 `/home/ikrfun/hf-env` はPython 3.12.3のみ確認でき、PyTorch・Transformers・Unsloth等は未導入。
- Gemma 4 E4Bの公式モデル識別子は `google/gemma-4-E4B-it`。Gemma 4はchat templateでthinkingを制御できるため、学習コードでテンプレートを手書きしない方針を継続する。

## 2026-07-27 — ikrfun-wsのuv環境構築

- 対象: `~/workspace/nasu-lora`
- `uv 0.11.32` を公式インストーラーで導入。
- Python 3.12.3の `.venv` を作成し、`uv add unsloth` で依存関係を解決。
- `pyproject.toml` と `uv.lock` を生成。
- 確認済み: PyTorch 2.11.0+cu130、Transformers 5.5.0、TRL 0.24.0、PEFT 0.19.1、Unsloth 2026.7.5。
- `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run ...` でRTX 5090（Compute Capability 12.0）を認識することを確認。
- 初期状態ではPyTorchのGPU番号がnvidia-smiと異なったため、学習コマンドではPCI Bus ID順と`CUDA_VISIBLE_DEVICES=0`を必ず指定する。
- 学習・モデルダウンロード・データ同期は未実施。

## 2026-07-27 — Gemma 4 E4Bロード・chat templateスモークテスト

- `google/gemma-4-E4B-it` のProcessorと `Gemma4ForConditionalGeneration` のロードに成功。
- 最初は `content` に文字列を渡して `TypeError` になった。Gemma 4では、公式例に合わせて `content` を `[{"type": "text", "text": "..."}]` の構造化形式にする必要がある。
- `processor.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` が成功し、入力15 tokenを確認。
- RTX 5090上で32 tokenの短い生成に成功。`processor.parse_response()` も成功し、日本語応答を取得。
- 学習コードでは、Gemma 4の構造化content形式と `enable_thinking=False` を使用する。

## 2026-07-27 — QLoRAスモーク学習

- 公式Gemma 4 E4B Text Notebookの構成を参考に、`scripts/train_qlora.py` と `configs/qlora-smoke.toml` を追加。
- `FastModel.from_pretrained(..., load_in_4bit=True, text_only=True)` で4bitベースをロード。
- `FastModel.get_peft_model` で言語層のみLoRA化。rank=8、alpha=8、dropout=0。
- 学習データ80件をGemma 4 chat templateで整形し、5090上で10ステップを完走。
- 条件: max_length=512、batch=1、gradient accumulation=4、learning rate=2e-4、seed=3407。
- 結果: 43.2秒、trainable parameters 18,350,080、train loss 5.3429。
- 出力: ikrfun-wsの `outputs/gemma4-e4b-qlora-smoke/` にadapter、tokenizer、`smoke-metrics.json` を保存。
- 学習後のGPUメモリは5090 48MiB、4090 15MiBまで解放された。
- 公式NotebookのTRL要件 `>=0.28.0` を試したが、インストール済みUnsloth 2026.7.5の依存条件 `trl<=0.24.0` と衝突したため、現環境では互換性のあるTRL 0.24.0を維持した。

## 2026-07-27 — QLoRA正式adapter（80件・1エポック）

- `configs/qlora-full.toml` を追加。80件、実効バッチ4のため `max_steps=20` として1エポックを明示した。
- 一度 `max_steps=-1` で実行したところTRLの既定3エポックになったため、途中で停止し、20ステップ設定へ修正して再実行した。途中出力は正式成果物として扱わない。
- 正式学習は20ステップ、1エポック、45.97秒で完走。
- 結果: `train_loss=4.1038`、`train_samples_per_second=1.74`。
- 出力: ikrfun-wsの `outputs/gemma4-e4b-qlora-full/adapter_model.safetensors`（約73MB）ほか設定・tokenizer・metrics。
- 正式adapterの再ロードと、那須のおすすめへの日本語生成に成功。RTX 5090を使用。
- 手動の `PeftModel` ラップではUnslothのGemma 4推論ラッパーと衝突したため、ロード後にGemma 4の`architectures`情報を補って生成する方法を確認した。

## 2026-07-27 — 固定評価データ20件の比較

- `scripts/evaluate_adapter.py` を追加し、同じ `eval.jsonl`・同じgreedy生成条件（max_new_tokens=64、thinking無効）で比較。
- ベースモデル: 那須/なす語を含む回答 6/20、🍆 1/20、平均121.55文字。
- 正式QLoRA adapter: 那須/なす語を含む回答 9/20、🍆 1/20、平均114.4文字。
- 結果JSON: ikrfun-wsの `outputs/eval-base.json` と `outputs/eval-qlora-full.json`。
- 20件の小規模評価なので品質の結論ではなく、口調・語彙変化の初期確認として扱う。adapter側では「なす」や那須への寄り道が増えた一方、NAS質問で反復が発生する回答もあり、追加学習やデータ調整時の確認対象とする。

## 次の実装

1. ~~ikrfun-wsに専用venvを作成し、RTX 5090向けのPyTorch、Transformers、Unsloth、TRL、PEFT、bitsandbytesを導入する。~~ 完了
2. ~~モデルの認証・ダウンロード可否と、tokenizerのchat template適用を最小スモークテストする。~~ 完了
3. ~~小規模なQLoRA試行を行い、loss・VRAM・生成結果を記録する。~~ 学習・adapter保存まで完了
4. 成功後にLoRA/QLoRA/DoRAを同一条件で比較する。

環境構築、モデルロード、QLoRAスモーク学習、正式1エポック学習、adapter再ロード生成まで完了。

## 2026-07-27 — 人手レビュー用候補1000会話

- `data/generate_draft_1000.js` を追加し、評価データを除外して`train.jsonl`の80件を種に候補を展開。
- `data/draft_1000.jsonl` を1000件生成。質問は1000件すべて異なり、各行に`draft_id`、`source_seed`、`variant`、`review_status`を付与。
- 草稿は機械的な質問・回答フレーム展開であり、正式学習には使わない。本人による約500件の選別・修正、重複除去、事実確認後に正式データへ採用する。
