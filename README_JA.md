
# Anima Regional Canvas（IMAGE入力対応）

元の `Anima Regional Canvas` をコピーし、赤・青などで塗られた `IMAGE` を直接受け取れるようにした専用拡張です。

- `image` が接続されている場合、その画像のサイズを自動的に使用します。
- 赤領域には `red_prompt`、青領域には `blue_prompt` を適用します。
- 黄・緑・マゼンタも元ノードと同じように認識します。
- `image` が未接続の場合は、元ノードと同じ手描きキャンバスとして使えます。
- 元の `Anima_Regional_Canvas` は変更しません。
- IMAGE接続中に `IMAGE → Paint` を押すと、接続IMAGEを編集可能なペイント面へ取り込みます。以後はペイント後の画像を領域判定に使用します。
- `Anima: SAM3 MASKを赤青IMAGE化` は、SAM3が出した人物MASK群から面積の大きい独立した2領域を選びます。
- `auto` では2人の重心差を見て左右または上下を判定し、左（または上）を赤、右（または下）を青にします。
- 赤青IMAGEの白背景は領域外です。赤と青が重なった場合も、境界で分割して重複を残しません。

## 接続

```text
赤青の領域IMAGE
  -> Anima Regional Canvas（IMAGE入力対応） image
```

SAM3と接続する場合:

```text
Load Image -> SAM3 Text Segmentation -> Anima: SAM3 MASKを赤青IMAGE化
                                      -> Anima Regional Canvas（IMAGE入力対応） image
```

`red_prompt` は赤MASKだけ、`blue_prompt` は青MASKだけに適用されます。IMAGE接続時は入力IMAGEの縦横サイズを自動採用するため、Canvasノードのwidth/heightは使いません。

`POSITIVE_TEXT` と `NEGATIVE_TEXT` はSTRING出力です。`POSITIVE_TEXT` は quality、scene、各色プロンプトの空でない項目を `, ` で結合します。既存のCONDITIONING出力 `POSITIVE` / `NEGATIVE` もそのまま残しています。

## 同梱ワークフロー

`workflows/SAM3_to_Anima_Regional_Canvas_Image_Input.json` をComfyUIへドラッグ＆ドロップしてください。画像読込、SAM3人物抽出、赤青変換、IMAGE入力対応Canvasまで接続済みです。

SAM3モデル本体は次の場所に必要です。

```text
ComfyUI/models/sam3/sam3.pt
```

現在の `comfyui-sam3` の `LoadSAM3Model` は、ファイルがなければHugging Faceから `1038lab/sam3` の `sam3.pt` を自動取得します。初回実行のみネット接続とダウンロード待ちが必要です。
