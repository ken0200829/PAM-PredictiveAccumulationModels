# Design: cue_locus — tie-anchored two-locus recovery-first design

- 文書版: 0.2.0
- 更新日: 2026-07-22
- Repo: `antovis86/PAM-PredictiveAccumulationModels`
- 状態: DRAFT（recovery manifest 凍結前）
- 関連仕様: `docs/cue_locus_behavioral_model_spec.md` v0.1.0

本designは、関連仕様の数式、データ契約、推定経路を変更せず、主要な回収対象を開始点 \(w\) とcoherence非依存drift bias \(v_0\) に縮約し、coherence依存gain \(v_g\) と表現アーキテクチャ比較を段階的に追加するための実行計画である。

## 1. 目的

37名のdot taskのchoiceとRTから、赤キューから学習された予測がDDMのどこへ作用するかを行動計算レベルで評価する。

主要な作用点は次の二つである。

1. 開始点 \(w\): 刺激提示前の境界バイアス
2. coherence非依存drift bias \(v_0\): 非tie試行の証拠蓄積中に持続する方向バイアス

coherence依存drift gain \(v_g\) は科学的に重要だが、主要な二作用点の回収可能性を確認した後の感度・昇格段階へ置く。目的は正の結果を得ることではなく、この実験設計と事前固定したモデル集合で、作用点をどこまで識別できるかを実データfit前に確定することである。

## 2. 変更しない契約

次の契約は `docs/cue_locus_behavioral_model_spec.md` から変更しない。

- 実装先は `analysis/pam_dot_task_python/` とする。
- PAM/TAPASに基づくPython joint-MAP経路を使用する。
- HGFとDDMの自由パラメータを同一objectiveで同時推定する。
- integratedモデルでは、承認済み案Dの白キュー用・赤キュー用の二本のeHGFを使い、HGFパラメータをcue間で共有する。
- parallelモデルでは、全刺激履歴に対するcue-blind eHGFを使う。
- OU belief軌跡を固定して再利用しない。知覚パラメータ、とくに \(\omega_2\) は新しいrecoveryでも生成・推定対象に含める。
- 全380試行をHGF更新へ使い、主要DDM尤度を試行101–380の有効応答に限定する。
- 学習期1–100のchoiceとRTは主要尤度へ入れず、conditional held-out・時間外挿PPCに使う。
- 反応期限とシミュレーション格子上限は3秒に固定する。
- tieではHGFを更新せず、DDMの `direction = 0`、従って \(v=0\) とする案Aを維持する。
- 一度生成した試行index付き事後予測バッチを、集約PPCと逐次PPCの双方で再集計する。
- 新モデルは別model ID、formulation version、config hash、run directoryで管理する。
- 現行 `ddm_w`、`ddm_w_c`、MATLAB fixtureの意味と値は変更しない。

HSSM、PyMC、NumPyro、固定OU軌跡、pooled LOOは本designの推定経路へ導入しない。これらを将来採用する場合は、親仕様と変更禁止境界を正式に改版し、PAM経路とは独立したモデル族として新しい回収を必要とする。

## 3. 過去のStage A結果の位置づけ

別プロジェクト `HGF_DDM` のStage Aは、本designの回収ゲートとして流用しない。Stage Aが検査したのは、固定OU beliefとHSSMを使った次の時間変化モデルである。

- static
- \(z\) couplingがテスト期に減衰するモデル
- \(v\)-gain couplingがテスト期に減衰するモデル

これは、本designの静的な `{history-only, w, v0}` や、parallel対integratedの識別とは異なる科学的問いである。したがって、Stage Aの結果は次の範囲でのみ利用する。

- beliefとDDM係数の積が弱識別になりうるという動機
- 大きなモデル集合を一度に実データへ適用せず、回収を先行させるという設計上の教訓
- gain候補を後段へ置く判断の感度根拠

Stage AのMAPスクリーナーは、その結果ファイルの契約どおりMCMC予算配分用であり、科学的モデル選択や構造的非同定の証拠として使用しない。現時点で許される表現は「HGF_DDMの特定のStage A条件では、\(v\)-gain減衰モデルを十分に回収できなかった」である。「gainは構造的に同定不能」とは結論しない。

## 4. 係数名と意味の分離

異なるプロジェクトで同じ記号が異なる意味を持つため、次の対応を固定する。

| 場所 | 係数 | 数学的意味 | 本designでの扱い |
|---|---|---|---|
| HGF_DDM | `b_w` | active cue-conditioned beliefによる \(z\) 変調 | 外部の参考結果。`gamma_w` と同一視しない |
| HGF_DDM | `b_v` | \(u\{a_v+b_v(belief-0.5)\}\) のbelief-dependent gain | 外部の参考結果。PAMの `b_v` と同一視しない |
| PAM integrated | `b_w` | 統合belief \(m_t\) による \(w\) 変調 | integratedの開始点作用 |
| PAM integrated | `b_v` | 非tieで \(b_v(m_t-0.5)\) に等しい加算的drift bias | integratedの \(v_0\) 作用 |
| cue parallel | `gamma_w` | raw cue evidence \(\kappa_t\) による開始点log-odds shift | parallelの開始点作用 |
| cue parallel | `gamma_v0` | raw cue evidence \(\kappa_t\) による非tie加算的drift bias | parallelの \(v_0\) 作用 |
| cue parallel | `gamma_vg` | 赤対白でcoherence slopeを変えるgain | 後段の感度候補 |
| cue integrated | `b_g` | 統合belief confidenceによるcoherence gain | 後段の感度候補 |

`gamma_w` とintegratedの `b_w` は、いずれも開始点を変えるが入力表現が異なる。同様に `gamma_v0` とintegratedの `b_v` は、いずれもcoherence非依存drift biasを表すが、前者は現在のraw cue evidence、後者はcueと履歴を統合したbeliefを入力とする。

## 5. 縮約するモデル集合

### 5.1 二作用点の候補

gainを除いた各アーキテクチャについて、次の四候補を使用する。

1. `history-only`: 新しい直接cue作用なし
2. `w`: 開始点作用のみ
3. `v0`: coherence非依存drift biasのみ
4. `w+v0`: 両作用を許す

`w+v0` を除外すると、両方の作用が存在する生成データを単一作用点へ強制分類してしまうため、縮約グリッドにも含める。

### 5.2 parallel

parallelではcue-blind eHGFの観測前予測 \(h_t\) と、反応境界へ整列した現在cue evidence \(\kappa_t\) を分けてDDMへ渡す。

開始点は

\[
w_t=\sigma\left[\operatorname{logit}\{0.5+b_{H,w}(h_t-0.5)\}+\gamma_w\kappa_t\right]
\]

とする。非tie試行のdriftは

\[
v_t=d_t\{a_v+b_c|x_t|\}+\gamma_{v0}\kappa_t
\]

とする。該当しない作用点の係数は0へ固定する。tieでは式を評価せず \(v_t=0\) とする。

候補model IDは次のとおりとする。

- `cue_history_w`
- `cue_parallel_w`
- `cue_parallel_vbias`
- `cue_parallel_w_vbias`

### 5.3 integrated

integratedでは、二本のcue別eHGFから得るactive cueの観測前予測 \(m_t\) だけをDDMへ渡し、raw \(\kappa_t\) をDDMへ重ねない。

開始点は

\[
w_t=0.5+b_w(m_t-0.5)
\]

とする。非tie試行のdriftは

\[
v_t=d_t\{a_v+b_c|x_t|\}+b_v(m_t-0.5)
\]

とする。tieでは \(v_t=0\) とする。

候補model IDは次のとおりとする。

- `cue_history_w`（parallelと共有するcue-blind history-only baseline）
- `cue_integrated_w`
- `cue_integrated_vbias`
- `cue_integrated_w_vbias`

親spec §6.3に従い、cue効果なしのbaselineは `cue_history_w` を両architectureで共有する。知覚信念をDDMへ接続しない自由なHGFパラメータはobjectiveへ含めない。

## 6. 段階的な回収計画

### Gate R0: 数式・配線

実データfit前に、parallelとintegratedの双方で次を通す。

- cue符号、choice境界、normal/reverse counterbalanceの解析的テスト
- 全model IDにおけるtieの \(v=0\)
- zero-effect nesting
- simulationとlikelihoodで同じtrialwise \(w,a,v,Ter\) を使用
- 1–100のchoice/RTが主要objectiveへ入らず、HGF更新には残る
- model IDと自由パラメータ集合の一致

### Gate R1: 各アーキテクチャ内の作用点回収

parallelとintegratedを別々の四候補集合として回収する。これにより、八候補を一度に競わせる前に、それぞれの表現内で `history-only`、`w`、`v0`、`w+v0` を区別できるか検査する。

生成条件は次のとおりとする。

- 実37名のtrial順、cue、coherence、condition、missingness mask
- HGF更新1–380、主要応答尤度101–380
- 3秒反応期限
- HGFとDDMを同時生成・joint-MAP回収
- \(\omega_2\)、Ter、基準drift、coherence slopeを含むnuisance truthの変動
- 作用点ごとに効果0、弱、中、強を含む
- 各主要生成セル20反復以上

異なる単位を持つ `gamma_w`、`gamma_v0`、`b_w`、`b_v` の効果水準を、係数値の単純な同倍率では定義しない。prior predictive simulationを用い、代表的な試行におけるcue-consistent choice確率差として弱・中・強を校正する。校正方法、代表試行、数値、seedを実データのモデル順位を見る前にrecovery manifestへ固定する。

実装上のprior候補 `cue-prior-candidate-0.2.0` では、実37名のtrial順、cue、coherence、condition、missingness maskだけを用い、観測choice/RT値を使わずに、cue-consistent choice確率差を弱0.005、中0.015、強0.025へ校正した。強効果の変換空間係数は `gamma_w=0.100083`、`gamma_v0=0.127215`、`b_w=2.276546`、`b_v=3.750562` である。これはrecovery用の候補値であり、model recoveryとprior感度を終えるまでは凍結済みGateとして扱わない。

### Gate R2: architecture回収

R1で両アーキテクチャが作用点を回収できた場合に限り、parallelとintegratedを同じ候補集合へ入れ、architecture×locusの混同行列を作る。

共有baseline `cue_history_w` は二つのarchitectureへ重複計上せず、architecture BMSでは `{history-only}`、`{parallel}`、`{integrated}` の三つの排他的familyを定義する。family priorを等確率、各family内のmodel priorを等確率とする主要集計に加え、全modelを一様priorとした場合の感度を報告する。

一方だけがR1を通過した場合、そのアーキテクチャ内の作用点についてのみ主張可能とし、parallel対integratedの優劣は主張しない。両方がR1を通過してもR2で混同する場合は、「作用点は識別できるが入力表現は識別できない」と報告する。

### Gate R3: gain感度

R1通過後にのみ、`cue_parallel_vgain` と `cue_integrated_vgain` を新しい生成・推定候補として追加する。過去のHGF_DDM結果をGate通過・不通過の代用にせず、本designの式、eHGF、joint-MAP、LME/BMSで新規回収する。

gainの回収が不十分なら「本実験と本モデル集合ではgain作用を識別できなかった」と報告する。数学的な構造的非同定を主張する場合は、少なくともtrialwise design matrixの相関・条件数、Hessian固有値、効果量とbelief変動幅を変えた回収を別途示す。

## 7. 一意な成功基準

本designではLOO+2SEを使用しない。主要なモデル比較は親specに従い、各被験者のLaplace LMEとrandom-effects Gibbs BMSで行う。

### 7.1 model recovery

各生成データ反復について、37名のLMEからBMSを実行する。中効果を「この研究で作用点を主張する最小目標効果」としてmanifestで指定し、事前固定した20反復以上に対し、中・強効果を主要ゲート、弱効果を検出力記述として次を判定する。

- 中・強効果でgenerating locus familyが最大expected frequencyとなる反復割合が、それぞれ80%以上
- null生成時にcue作用ファミリーを選ぶ偽陽性率が10%以下
- `w` 生成を `v0` とする割合、`v0` 生成を `w` とする割合がそれぞれ10%以下
- `w+v0` 生成時に単一作用点へ縮退する割合を独立に報告
- generating exact modelが最大LMEとなる被験者割合と、最大expected frequencyとなるモデルを独立に報告
- 効果量別の分類率を報告し、弱効果の不確実性と強効果での混同を区別

architecture比較では、上記のlocus基準とは別に、中・強効果でgenerating architecture familyが最大expected frequencyとなる反復割合80%以上を要求する。locusとarchitectureのどちらか一方だけが通過した場合、通過した軸についてのみ主張する。

### 7.2 parameter recovery

親specと既存 `recovery-criteria-1.0.0` に従う。

- 真値対推定値相関 \(r\ge0.70\)
- 変換空間の絶対bias \(\le0.50\)
- RMSE / prior SD \(\le1.0\)
- 各自由パラメータ20ケース以上

特に `gamma_w`–`gamma_v0`、`b_w`–`b_v`、\(\omega_2\)–各DDM傾きの推定相関、Hessian条件数を保存する。

回収結果を見た後に閾値、候補集合、prior、効果水準を変更しない。変更が必要な場合はmanifestを改版し、旧反復を新ゲートへ混在させず、全候補を同じ版で再実行する。

## 8. 階層化の扱い

現行PAM経路は被験者別joint-MAP、Laplace LME、集団random-effects BMSであり、HSSMの部分プーリングを含まない。本designではStage Hを回収改善策として追加しない。

別プロジェクトのpooled Stage Aは37名のcomplete poolingであり、独立被験者推定ではない。部分プーリングは被験者差を表現するための別の推定モデルであり、complete poolingより回収率を必ず改善するものではない。

将来hierarchical DDMを検討する場合は、次を別仕様で事前固定する。

- 推定ライブラリと尤度
- \(a\)、Ter、基準drift、coherence slope、cue係数のrandom-effects構造
- 生成側と回収側で同じ階層構造を使う自己回収
- 個人差を持つnuisanceを固定した場合の誤指定感度
- PAM LME/BMS結果との比較可能範囲

## 9. tie診断

案Aの下ではtieの \(v=0\) なので、cue別choice差は開始点経路を診断する。ただしnormalとreverseをrawな白判断率のままプールしない。

被験者 \(i\) の符号整列したtieコントラストを

\[
\Delta_{tie,i}=r_i\left\{P_i(white\mid red,tie)-P_i(white\mid white,tie)\right\}
\]

とする。\(r_i=+1\) は赤が白を予測する条件、\(r_i=-1\) は赤が黒を予測する条件である。正の \(\Delta_{tie,i}\) は、赤キューが予測境界へのchoiceを増やしたことを表す。

次を報告する。

- 被験者別 \(\Delta_{tie,i}\)
- 37名平均と被験者単位bootstrap信頼区間
- normal/reverse別の未変換率と符号整列後の率
- cue別・choice-conditional RT
- 生成truthと \(\Delta_{tie}\) の単調対応を確認するrecovery PPC

差が観測されないことだけから \(w=0\) とは結論しない。回収で定めた最小識別可能効果を等価限界として事前固定できた場合に限り、等価性検定を補助的に行う。

この生データ図は、数式、候補モデル、prior、recovery manifestを凍結した後に記述統計として作成する。図の結果によって主要モデル集合やゲートを変更しない。凍結前に結果を見て設計を変更した場合、その後の解析は探索的と明記する。

## 10. PPC

Gate通過後の実データ解析では、親specの集約PPCと逐次PPCを実行する。一度の事後予測生成で試行位置を保持し、次を窓別に再集計する。

- cue×signed coherence別の白判断率
- cue×signed coherence別のRT q10/q50/q90
- 符号整列したtie choiceコントラストとchoice-conditional RT
- 速い・中間・遅いRT分位帯のcue効果
- test試行位置別・cue提示順別の時間分解統計量
- 試行1–100のconditional held-out PPC

窓ごとに新しい予測標本を生成しない。観測値、予測中央値、同時予測帯、窓内有効試行数を保存する。

## 11. 実装順序と停止規則

1. 本designと親specの整合を確定する。
2. `cue_evidence = cue_red * red_prediction_sign`、`choice_white`、`is_tie` の監査列とテストを実装する。
3. parallel/integratedの純粋なtrialwise関数とmodel IDを実装する。
4. zero-effect nesting、tie、対称性、simulation-likelihood整合テストを通す。
5. prior predictive simulationでpriorと弱・中・強の効果水準を校正する。
6. 候補集合、反復数、seed、成功基準を含むrecovery manifestを凍結する。
7. Gate R1を実行する。不通過なら実データの作用点比較へ進まない。
8. R1通過範囲に応じてGate R2を実行する。architecture不通過ならP対Iを主張しない。
9. 凍結後に符号整列した実データtie記述図を作成する。
10. 3–5名の非報告スモークfitと逐次PPCを行い、配線と数値安定性だけを確認する。
11. model registryを最終凍結し、37名本解析、BMS、PPCを行う。
12. R1通過後、必要な場合だけGate R3のgain感度を別manifestで実行する。

停止規則は次のとおりである。

- R1で作用点を回収できない: cue効果の有無は検討できても、\(w\) 対 \(v_0\) を主張しない。
- R1は通るがR2が通らない: 作用点についてのみ主張し、parallel対integratedを区別しない。
- gainが回収できない: gainを主要結論へ含めず、当該設計での識別限界として報告する。
- 数値失敗や欠損LMEがモデル間で偏る: 欠損値を補完せず、原因を解消するまでBMSへ進まない。

## 12. 依存関係と成果物

依存するのは現在の `analysis/pam_dot_task_python/` の次の経路である。

- `data.py`
- `hgf.py`
- `response.py`
- `config.py`
- `objective.py`
- `recovery.py` / `gates.py`
- `ppc.py`
- Laplace LMEとGibbs BMS

新たな必須依存としてHSSM/PyMC/NumPyroを追加しない。

凍結・保存する成果物は次のとおりである。

- formulation、prior、mask、recovery、PPCのversionとhash
- 生成truthと実37名のdesign/missingness契約
- 被験者×モデルのMAP、Hessian、LME、収束診断
- locusおよびarchitectureのmodel-recovery混同行列
- parameter-recovery指標と推定相関
- 符号整列したtie診断
- 集約PPCと逐次PPC
- Gate判定と、通過しなかった軸に関する主張制限

## 13. 実装開始条件

実装開始前に未確定であってよいのは、prior predictive simulationで校正する効果水準の数値だけである。その数値は実データのchoice/RTやモデル順位を見ずに決め、recovery manifestへ固定する。

次の項目は本designで確定済みとする。

- 推定経路: PAM Python joint-MAP + Laplace LME + random-effects Gibbs BMS
- 主要尤度: 試行101–380
- tie: \(v=0\)
- 初期作用点集合: `history-only`、`w`、`v0`、`w+v0`
- architectureの順序: 各architecture内回収後にcross-architecture回収
- gain: R1後の別manifestによる感度・昇格候補
- hierarchical HSSM: 本designの対象外
- 実データtie図: recovery manifest凍結後の記述診断

以上を満たした時点で、Gate R0の実装へ進む。
