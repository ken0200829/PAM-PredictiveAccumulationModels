# キュー作用点を行動データから識別するモデルの実装仕様

- 文書版: 0.1.0
- 作成日: 2026-07-22
- 対象: 37名 dot task の Python 解析層
- 実装先: `analysis/pam_dot_task_python/`
- 状態: 実装前仕様。数式とデータ契約は固定し、新規 prior と recovery manifest は実データのfit結果を見る前に版付きで凍結する。

## 1. 目的と主張可能範囲

本実装の目的は、赤キューから学習された予測が選択と反応時間へ及ぼす作用を、次の二軸で比較することである。

1. **表現形式**
   - 独立並行表現: 刺激履歴から得る動的信念と、現在のキューが与える即時的情報をDDMへ別々に入力する。
   - 共通確率表現: キューと履歴から得る予測を知覚層で一つの白カテゴリ予測確率に統合し、その確率だけをDDMへ渡す。
2. **DDM内の作用点**
   - 相対的開始点 \(w\)
   - coherence非依存のdrift bias \(v_0\)
   - coherence依存のdrift gain \(v_g\)

行動データから主張するのは「どの計算モデルがchoiceとRTの同時分布を最もよく説明したか」である。勝ったモデルを神経機構そのものの証明とは解釈しない。論文では、例えば「行動はキューが蓄積開始前のバイアスを変えるモデルと最も整合した」と記述する。

## 2. 既存実装との関係

本実装は、以下の既存仕様を変更しない。

- PAM/TAPASに基づくPython joint-MAP経路
- 案Dの外部カスタムperceptual model
- 白キュー用・赤キュー用の二本のeHGF、cue間で共有するHGFパラメータ
- 非active cueの状態凍結とcue提示回数軸
- 3秒の反応期限
- tieでHGFを更新せず、DDMでは `direction = 0`、従って \(v=0\) とする案A
- 全380試行を知覚更新へ使用し、主要な応答尤度を試行101–380に限定する規約
- 一度生成した試行index付き事後予測バッチを集約PPCと逐次PPCで再利用する規約

現行 `ddm_w` と `ddm_w_c` は参照モデルとして保存し、意味を変更しない。現行 `ddm_w` はactive-cue HGFの予測が \(w\) だけを変えるモデルであり、本仕様の「直接キュー項を持つparallelモデル」ではない。新モデルはすべて別のmodel ID、config hash、run directoryで管理する。

数式変更後の新モデルはPythonのみで実装する。既存MATLAB fixtureは旧モデルの回帰テストとして保持するが、新数式の正当性を旧fixtureとの一致で判定しない。新数式は解析的テスト、生成・尤度整合、parameter recovery、model recoveryで検証する。

## 3. 実験・データ契約

### 3.1 実際のキューの意味

提示されるキューは赤い十字と白い十字の二種類であり、左予測・neutral・右予測の三水準キューではない。白キューは学習期の刺激分布が白黒について対称な参照条件であり、赤キューだけが学習期に強いカテゴリ予測を与える。

解析上の0は「画面にneutral cueが提示された」ことを意味しない。白キューを操作上の参照値0として符号化した結果である。

### 3.2 反応境界に合わせた符号

すべての条件で上側境界を白判断、下側境界を黒判断とする。F/Jキーのカウンターバランスは `choice_white` 作成時に解消する。

被験者 \(i\)、試行 \(t\) について、赤キューが学習期に予測した表示カテゴリの符号を

\[
r_i=
\begin{cases}
+1,&\text{normal または normal\_cb（赤が白を予測）},\\
-1,&\text{reverse または reverse\_cb（赤が黒を予測）}
\end{cases}
\]

とする。また、赤キュー提示を

\[
q_{it}=\mathbb I(\text{cue}_{it}=\text{red})
\]

とし、方向を持つ即時キュー証拠を

\[
\kappa_{it}=r_i q_{it}
\]

と定める。従って \(\kappa\) は赤試行でのみ \(+1\) または \(-1\)、白試行では0である。これはcue色のraw codingではなく、白／黒の反応境界へ整列した学習 contingency の符号である。一方、\(q\) は方向を持たない赤対白のcue contrastであり、coherence gainの比較に使う。固定倍率を掛けても係数へ吸収されるため、学習期の0.9/0.1をそのまま係数へ埋め込まず単位化する。

赤キューは学習期に「特定カテゴリ」と「高coherence」の両方を予測する。従って \(\kappa\) は方向性のある開始点／drift bias仮説、\(q\) は方向性のないgain仮説に使い分ける。ただし各被験者内では \(\kappa_{it}=r_iq_{it}\) であり、二つは独立に操作されていない。方向biasとgainの識別は、主にtie試行とcue×coherenceの形状、およびmodel recoveryに依存する。

実装では次を必ず監査列へ保存する。

- `cue_red`
- `red_prediction_sign`
- `cue_evidence = cue_red * red_prediction_sign`
- `choice_white`
- `ratio_corrected`
- `signed_coherence`
- `is_tie`

`red_prediction_sign` はファイル名から推測したキー配置ではなく、検証済みの `condition` と `STIMULUS_REVERSED` 契約から作る。反応キーカウンターバランスはこの符号へ二重に適用しない。

### 3.3 刺激の符号

\[
x_t=2\,\mathtt{ratio\_corrected}_t-1,
\qquad
T_t=\mathbb I(x_t=0)
\]

とする。非tie試行では

\[
s_t=\mathbb I(x_t>0),
\qquad
d_t=2s_t-1=\operatorname{sign}(x_t)
\]

である。tieでは \(s_t\) を概念上未定義とし、placeholder値を計算へ使わない。全response modelは `is_tie` で明示分岐し、tieでは必ず \(d_t=0,\ v_t=0\) を返す。

### 3.4 試行1–100と101–380

- 試行1–100: HGFの学習入力に含める。主要DDM応答尤度からは除外する。
- 試行101–380: HGFを継続更新し、有効なchoice/RTを主要DDM応答尤度へ含める。
- 試行1–100の観測choice/RT: conditional held-out・時間外挿PPCにだけ使う。

試行1–100も応答尤度へ含める解析は感度分析として別のlikelihood-mask IDでのみ許可する。主要解析と試行数が異なるため、主要解析のLME/BMS行列へ混在させない。

## 4. 共通DDM部分

境界幅と非決定時間は主要モデル間で同じとする。

\[
a_t=a_a,
\qquad
Ter_t=Ter
\]

主要な作用点比較では \(b_a=0\) に固定する。境界変調を同時に自由にすると \(w\) と \(v\) の識別を不要に難しくするためである。既存PAMの \(b_a\) を含むモデルは、主要な \(w\) 対 \(v\) 比較が回収可能と確認された後の感度分析とする。

感覚証拠の共通部分を

\[
v_{\mathrm{sens},t}=d_t\{a_v+b_c|x_t|\}
\]

とする。\(a_a>0\)、\(a_v>0\)、\(0<Ter<\min(RT_{valid})\) は既存変換を維持する。`b_c` はcoherence主効果であり、新しいキューgain係数とは別名にする。

## 5. アプローチP: 独立並行表現

### 5.1 知覚層

キューを入力に使わない単一のcue-blind eHGFを、全刺激履歴に対して走らせる。試行 \(t\) の刺激を観測する前の白カテゴリ予測を

\[
h_t=P(s_t=1\mid s_{<t})
\]

とする。tieでは既存案Aと同じく更新を行わない。全380試行を通じて一つのglobal trial軸を使う。

この \(h_t\) は過去の刺激系列から得た動的信念、\(\kappa_t\) は現在の赤／白キューから得た即時情報である。両者を知覚層で足し合わせず、DDMへ別々に入力する。

### 5.2 開始点

まず履歴だけによるPAM互換開始点を

\[
w^{(H)}_t=0.5+b_{H,w}(h_t-0.5),
\qquad b_{H,w}\in(-1,1)
\]

とする。直接キュー効果を持つモデルでは

\[
w_t=\sigma\{\operatorname{logit}(w^{(H)}_t)+\gamma_w\kappa_t\}
\]

とする。`gamma_w = 0` なら \(w_t=w^{(H)}_t\) に厳密に戻る。単純加算後のclipは禁止する。logit上で加算することで、全試行で \(0<w_t<1\) を保証し、境界への張り付きも診断可能にする。

### 5.3 drift biasとdrift gain

非tie試行では

\[
v_t=
v_{\mathrm{sens},t}
+\gamma_{v0}\kappa_t
+d_t\gamma_{vg}|x_t|q_t .
\]

- `gamma_v0`: coherenceの大きさによらず、赤キューが予測した境界へdriftを押す即時バイアス
- `gamma_vg`: 赤対白でcoherence感度を変える、カテゴリ方向に対して対称なgain

最後の項は \(\gamma_{vg}x_tq_t\) と等しい。従って `gamma_vg` は赤キュー下でpsychometric functionの傾きを変えるが、白／黒のどちらか一方へ一定方向に押す項ではない。

tieでは上式を評価せず、確定仕様として

\[
v_t=0
\]

とする。従って本研究の `gamma_v0` は「非tie試行における加算的drift bias」であり、一般的な線形DDMのようにtieでもdriftを生じさせる定式化ではない。この制約は案Aを維持するための理論的仮定であり、論文に明記する。

### 5.4 parallelモデルID

| model ID | 自由なキュー作用 | 用途 |
|---|---|---|
| `cue_history_w` | なし | 全候補で共有するcue-blind履歴ベースライン |
| `cue_parallel_w` | `gamma_w` | cue→開始点 |
| `cue_parallel_vbias` | `gamma_v0` | cue→非tie drift bias |
| `cue_parallel_vgain` | `gamma_vg` | cue→coherence依存gain |
| `cue_parallel_w_vbias` | `gamma_w`, `gamma_v0` | 開始点＋drift bias |
| `cue_parallel_w_vgain` | `gamma_w`, `gamma_vg` | 開始点＋gain |

全モデルで `b_H,w` と `b_c` を同じ規約で自由にする。三つのキュー作用を同時に自由にするfullモデルは、上記縮約モデルのmodel recoveryを通過した場合だけ探索的に追加する。

## 6. アプローチI: 共通確率表現

### 6.1 知覚層

承認済み案Dの二本のcue別eHGFをそのまま用いる。active cueについて、試行 \(t\) の観測前予測を

\[
m_t=P(s_t=1\mid \text{cue}_t,\text{同じcueの過去履歴})
\]

とする。\(m_t\) はキューと履歴が既に統合された一つの白カテゴリ確率であり、DDMへraw cue項 \(\kappa_t\) を別途渡さない。

当初案にあった

\[
\tilde m_t=\sigma\{\operatorname{logit}(h_t)+\beta_{cue}\kappa_t\}
\]

は主要実装に採用しない。下流の \(b_w\) または \(b_v\) と `beta_cue` が積として振る舞い、現在の試行数では弱識別になりやすいこと、および案Dが既に同じ科学的役割をcue条件付き予測確率として実装しているためである。この式を将来検討する場合は別のperceptual model IDと独立したrecovery gateを必要とする。

### 6.2 DDMへの写像

開始点は

\[
w_t=0.5+b_w(m_t-0.5),
\qquad b_w\in(-1,1)
\]

とする。統合beliefの確信度様変調量を、既存PAMと同じく

\[
\phi_t=
\sigma\left\{\frac{1}{m_t(1-m_t)}-4\right\}-0.5
\]

とする。非tie試行のdriftは

\[
v_t=
v_{\mathrm{sens},t}
+b_v(m_t-0.5)
+d_tb_g|x_t|\phi_t
\]

とする。ここで既存PAMの

\[
d_t b_v\{p_{\mathrm{presented},t}-0.5\}
\]

は、非tie試行では \(b_v(m_t-0.5)\) と代数的に等しい。このため `b_v` は予測カテゴリ方向へのcoherence非依存drift biasとして解釈できる。`b_g` は統合beliefの確信度 \(\phi_t\) に応じて白／黒に対称なcoherence感度を変える新しいgainである。

tieではparallelモデルと同じく

\[
v_t=0
\]

とし、\(w_t\) だけは通常どおり \(m_t\) から計算する。

### 6.3 integratedモデルID

| model ID | 自由なbelief作用 | 既存モデルとの関係 |
|---|---|---|
| `cue_integrated_w` | `b_w` | 現行 `ddm_w_c` と数式同一。別名aliasではなく既存IDを参照可能 |
| `cue_integrated_vbias` | `b_v` | 現行 `ddm_v_c` と数式同一 |
| `cue_integrated_vgain` | `b_g` | 新規 |
| `cue_integrated_w_vbias` | `b_w`, `b_v` | `ddm_full_c` から `b_a=0` とした新しい縮約 |
| `cue_integrated_w_vgain` | `b_w`, `b_g` | 新規 |

既存 `ddm_full_c` は `b_a` も自由であるため、主要な `cue_integrated_w_vbias` と同一視しない。

cue効果なしの共有ベースラインには `cue_history_w` を使う。知覚信念を一切DDMへ接続しない `ddm_c` は感覚・運動のみの補助nullとして残すが、自由なHGFパラメータを持たせない。これにより、DDMへ接続されないHGFパラメータをjoint objectiveへ含める非識別モデルを作らない。

## 7. tie試行が与える識別情報

案Aの下では全モデルでtieの \(v=0\) である。従ってtieのcue別choice差は開始点経路だけから生じる。

- `*_w`: tieでcue別の白判断率が変わりうる。
- `*_vbias`, `*_vgain`: tieでcue別choice差を作れない。
- `*_vgain`: 非tieでcueまたはbelief confidence×|signed coherence|の傾き差を作る。
- `*_vbias`: 非tieでcoherenceに依存しないcue方向のdrift差を作る。

この識別は \(v=0\) という仮定に条件づく。tieでもキューdriftを許す線形DDMを同じモデル名へ混ぜない。将来それを検討する場合は `tie_drift_allowed` を含む別のformulation versionとする。

## 8. パラメータ変換とprior

legacy modelのパラメータ変換とpriorは現行実装を維持する。cue-locus候補では、異なる係数単位がmodel evidenceを不公平に変えないよう、変換空間で平均0の対称Gaussian priorを共通のchoice確率効果尺度へ校正する。

| パラメータ | native-space意味 | 変換・support | `cue-prior-candidate-0.2.0` |
|---|---|---|---|
| `b_H_w` | cue-blind historyによる開始点傾き | (-1, 1) | Normal(0, 2²)（legacy維持） |
| `gamma_w` | cueによる開始点log-odds shift | 無制約 | Normal(0, 0.051064²) |
| `gamma_v0` | cueによる非tie drift bias | 無制約 | Normal(0, 0.064907²) |
| `b_w`（integrated） | 統合beliefによる開始点傾き | (-1, 1) | Normal(0, 1.161525²) |
| `b_v`（integrated） | 統合beliefによるdrift bias | 無制約 | Normal(0, 1.913587²) |
| `gamma_vg` | cueによるdrift gain | 無制約 | Normal(0, 2²) |
| `b_g` | 統合beliefによるdrift gain | 無制約 | Normal(0, 2²) |

主要4係数は、実37名の結果を伏せた設計とmissingness mask上で、cue-consistent choice確率差0.005、0.015、0.025を弱・中・強と定義して校正した。各係数の強効果値が対称Normal priorの97.5 percentile（中央95%）になるようSDを定めた。`gamma_vg` と `b_g` はGate R3まで昇格しないため初期値のままである。

prior predictive監査は旧PAM nuisance priorの再設計ではなくcue作用priorの校正を目的とするため、HGF、基準drift、境界、coherence slope、Terはprior平均に固定し、各modelのcue response effectsだけをサンプルする。固定seed 202607221、16 draws×4反復、10 ms格子、3秒上限で4条件×parallel/integratedを監査する。次を満たした候補だけをrecoveryへ渡す。

- \(w\) がほぼ0または1に張り付く試行の割合
- 3秒以内の応答確率 `captured_mass`
- choice率とRT分位点の現実的範囲
- prior変更に対するモデル順位の感度

結果を見た後にprior幅をモデルごとに変えない。同じ種類の係数には同じpriorを使う。

現在の候補は全8監査セルで数値失敗0、`w<0.01` または `w>0.99` の割合0、median `captured_mass` 約1.000、生成choice率約0.50、生成RT中央値0.47–0.48秒で上記の工学的監査を通過した。ただしmodel recoveryとprior変更に対する順位感度が未完了なので、statusは `candidate_not_frozen` のままとする。

## 9. 推定とモデル比較

### 9.1 個人推定

- HGFとDDM自由パラメータを同一のjoint objectiveでMAP推定する。
- 主要likelihood maskは試行101–380の有効応答のみとする。
- optimizer、Hessian、Laplace LME、Ter診断は既存経路を再利用する。
- 各被験者×モデルで複数の事前固定初期値を実行し、有限な最小negative log jointを採用する。
- model ID、formulation version、prior hash、mask hash、seedをrun manifestへ保存する。

### 9.2 集団比較

主要比較は、全候補モデルが正常終了した被験者の共通集合に対するindividual LMEとrandom-effects BMSで行う。欠けたLMEを補完しない。

二つのモデルファミリー比較を事前定義する。

1. 表現ファミリー: `parallel` 対 `integrated`
2. 作用点ファミリー: `history-only`、`w`、`vbias`、`vgain`、`w+vbias`、`w+vgain`

ファミリー内のモデル数が異なる場合は、model frequencyを単純加算するだけでなく、モデル数の不均衡に対する感度分析を併記する。protected exceedance probabilityは感度分析、論文忠実なGibbs BMSを主解析とする既存規約を維持する。

### 9.3 解釈規則

- `w`ファミリーが優位: 行動は刺激前の開始点変調と整合する。
- `vbias`ファミリーが優位: 行動は非tie試行で継続する方向性drift biasと整合する。
- `vgain`ファミリーが優位: 行動はcoherence依存の感覚証拠変調と整合する。
- `w+v`が優位: 単一作用点では不十分。ただし両係数の回収と事後相関を必須報告する。
- 回収不能: cue効果の存在は論じても、その作用点を一意に決めない。

## 10. model recoveryとparameter recovery

新モデルのfitを実データへ適用する前に、新しいversionのrecovery manifestを作る。既存V3.1の結果を新モデルのGate通過として流用しない。

### 10.1 必須の生成条件

- 実際の37名のtrial順、cue、coherence、condition、missingness maskを使用する。
- 反応期限は3秒に固定する。
- HGF更新は1–380、主要応答尤度は101–380とする。
- history-only、\(w\)、\(v_0\)、\(v_g\)、\(w+v_0\)、\(w+v_g\) をそれぞれ生成モデルにする。感覚・運動のみの `ddm_c` nullも偽陽性検査用に含める。
- parallelとintegratedをそれぞれ生成側・推定側へ置き、architecture×locusの混同行列を作る。
- 効果0、弱、中、強の範囲を含め、生成truth同士の相関を避ける。
- `omega_2`、Ter、基準drift、coherence slopeもtruth gridで変化させる。

### 10.2 parameter recovery基準

既存 `recovery-criteria-1.0.0` を最低基準として維持する。

- 真値対推定値相関 \(r\ge0.70\)
- 変換空間の絶対bias ≤0.50
- RMSE / prior SD ≤1.0
- 各自由パラメータ20ケース以上

特に `gamma_w`–`gamma_v0`、`gamma_w`–`gamma_vg`、`b_w`–`b_v`、`b_w`–`b_g`、\(\omega_2\)–各DDM傾きの推定相関を保存する。

### 10.3 model recovery基準

初回生成前に、反復数、効果量水準、選択指標、合格率をmanifestへ固定する。最低限、次を独立に報告する。

- generating modelが最大LMEとなる割合
- generating locus familyがBMSで最大expected frequencyとなる割合
- \(w\rightarrow v\) と \(v\rightarrow w\) の誤分類率
- null生成時の偽陽性率
- 効果量別の分類率

弱い効果での不確実性を、強い効果での構造的混同と区別する。合格しない場合は結果後に閾値を緩めず、候補モデルを統合するか、論文の主張を「作用点は識別不能」へ狭める。

## 11. 事後予測チェック

既存 `gate-ppc-1.0.0` の2000反復、3秒格子、Laplace/MAP退避規則と逐次PPCを再利用できる。ただし新しいtrialwise量を予測バッチと監査出力へ追加する。

- global HGF予測 \(h_t\)
- active-cue HGF予測 \(m_t\)
- cue contrast \(q_t\)
- cue evidence \(\kappa_t\)
- \(w_t,a_t,v_t,Ter_t\)
- driftの内訳: sensory、cue bias、cue gain、belief bias、belief gain

主要PPCは次を含む。

1. cue×signed coherence別の白判断率
2. cue×signed coherence別のRT q10/q50/q90
3. tie試行のcue別白判断率とchoice-conditional RT
4. 速い・中間・遅いRT分位帯におけるcue効果
5. test試行位置別、cue提示順別の逐次PPC
6. 試行1–100のconditional held-out PPC

一つの事後予測生成バッチを試行index付きで保存し、上記の窓ごとに再集計する。窓ごとに再生成しない。

## 12. 実装単位

### 12.1 コード

- `data.py`
  - `red_prediction_sign` と `cue_evidence` を生成・監査する。
- `hgf.py`
  - cue-blind global HGFと既存two-cue HGFを明示的に別APIで返す。
- `response.py`
  - parallel/integratedのtrialwise \(w,a,v,Ter\) と成分分解を実装する。
  - tie分岐を共通関数化し、placeholder `stimulus_category` を参照しない。
- `config.py`
  - 新model ID、自由パラメータ集合、変換、priorを登録する。
  - model IDと自由パラメータ集合の一致を自動検査する。
- `objective.py`
  - architectureに応じてglobal HGFまたはtwo-cue HGFを選び、同一joint objectiveへ接続する。
- `ppc.py`
  - simulationとlikelihoodが同じtrialwise関数を使用する。
- `gates.py` / `manifests/`
  - 新しいprior freeze、parameter recovery、model recoveryを版付きで保存する。

### 12.2 必須テスト

1. `cue_evidence` がnormalで赤=+1、reverseで赤=-1、白=0になる。
2. F/J counterbalanceを変えても `cue_evidence` は変わらない。
3. `gamma_w = 0` でparallel開始点がhistory-only開始点へ厳密に戻る。
4. 全試行で \(0<w<1\)。clipを使わない。
5. 全model IDでtieの \(v=0\)。
6. 非tieで既存 `b_v` 式と \(b_v(m-0.5)\) が数値的に一致する。
7. `gamma_vg` 項が白黒カテゴリに対称で、赤対白のcoherence slopeだけを変える。
8. 新規効果を0にすると宣言された親モデルへ試行別log likelihoodが機械精度内で戻る。
9. 白黒カテゴリ、normal/reverse、上下境界を同時反転したときの対称性が保たれる。
10. simulationとlikelihoodで同じ \(w,a,v,Ter\) が使われる。
11. 1–100のchoice/RTが主要objectiveへ入らず、HGF更新とheld-out PPCには残る。
12. model名と自由パラメータ集合が不一致なら停止する。
13. recovery manifestのdigestが変われば旧結果を再利用できない。

## 13. 保存物

被験者×モデルごとに最低限、以下を保存する。

- 入力監査表とlikelihood mask
- transformed/native MAP、gradient、収束理由、初期値別結果
- trialwiseの \(h_t\) または \(m_t\)、\(q_t\)、\(\kappa_t\)、\(w,a,v,Ter\) とdrift成分
- negative log likelihood、negative log joint、LME、AIC、BIC
- 数値Hessian、共分散、相関、固有値、条件数、Ter診断
- aggregate/逐次PPCと単一のposterior-prediction batch参照
- model/prior/recovery/PPC specificationのversionとhash

集団出力としてLME行列、BMS入力対象者、除外理由、model/family posterior、model-recovery混同行列を保存する。

## 14. 実装順序と停止ゲート

1. データ監査列とcue符号テスト
2. parallel/integratedの純粋なtrialwise関数と解析的テスト
3. config・objective・simulationへの接続
4. zero-effect nestingと生成・尤度整合
5. prior predictive simulationとprior manifest凍結
6. parameter recoveryとmodel recoveryのmanifest凍結
7. recovery実行
8. 3–5名の非報告スモークと逐次PPC
9. model registryの最終凍結
10. 37名本解析、BMS、PPC

Gate 7で作用点またはarchitectureが回収できなければ、37名結果からその区別を主張しない。Gate通過前の実データfitは配線・数値診断に限り、モデル順位や効果量を報告値として使用しない。

## 15. 論文で必ず明記する限界

- 神経指標を測定していないため、\(w\) または \(v\) の優位性は行動計算レベルの結論である。
- 白キューは表示上のneutral cueではなく、対称な学習履歴を持つ参照キューである。
- 赤キューのカテゴリ方向と期待coherenceは学習期に同時操作されており、方向biasとgainの分離はモデル仮定とテスト期のcoherence交差に依存する。
- tieで \(v=0\) とする結論は案Aに条件づき、tie driftを許す一般的線形DDMとは異なる。
- 主要DDM尤度はテスト期に限定され、学習期PPCはconditional held-out・時間外挿診断である。
- 37名で個人差パラメータが弱識別なら、個人値ではなく集団モデル頻度と回収可能な係数だけを解釈する。
- model recoveryが不十分なら、キュー効果の存在と、その作用点の識別を分けて報告する。
