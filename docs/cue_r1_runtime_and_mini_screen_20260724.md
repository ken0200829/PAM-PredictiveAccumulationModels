# Gate R1 の実行時間と小規模事前診断

## なぜ「HGFモデル回収」が数週間になるのか

本プロジェクトの Gate R1 は、HGFだけを1回推定する処理ではない。HGFとDDMを同じ
目的関数で同時推定し、各生成データに対して候補4モデルを3初期値からMAP推定した後、
数値HessianとLaplace LMEを計算する集団model-recoveryである。

凍結済みの全R1は次の規模を持つ。

- 20生成セル × 20反復 × 37名 = 14,800 subject-task
- 各taskで4候補をfitするため、59,200 LME fit
- 主要14セルだけでも10,360 subject-task、41,440 LME fit
- 各fitは3初期値なので、主要部だけで最大124,320 MAP最適化に加えてHessian計算
- HGFは380試行を更新し、DDM尤度は試行101–380の有効応答について反復評価される

既存のHGFキャッシュ有効タスク363件では、4候補を含む1 subject-taskの実測中央値が
164.5秒、平均が204.0秒だった。したがって主要部全体は、単純換算で1並列なら
約19.7–24.5日、4並列でも約4.9–6.1日を要する。キャッシュ導入前81件の中央値は
855.7秒だったため、当初の1並列見積もりが3–6週間になったこととも整合する。

友人の「1–2時間」と両立する典型的な違いは、単一モデル・単一または少数の生成条件、
少数の被験者または疑似被験者、1初期値、MAPだけ、固定HGF軌跡、あるいは
parameter recoveryだけを指している可能性である。本R1は、作用点、入力表現、
集団BMS、nuisance変動まで同時に検査するため、同じ「モデル回収」という呼称でも
計算単位が異なる。

## 1–2時間の mini screen

正式R1を変更せず、別バージョン `cue-r1-mini-screen-0.1.0` を追加した。

- 4 counterbalance条件から各2名、合計8名
- 選択にはconditionと有効試行数だけを使い、choice/RT値は使わない
- 各architectureで null、w-medium、v0-medium の3生成セル
- 各セル2反復
- fit側は正式R1と同じ4候補、3初期値、joint MAP、Laplace LME
- HGFキャッシュ256、4並列
- 合計96 subject-task、384 LME fit

既存実測からの所要時間は約1.1–1.4時間である。これは「gross
non-recoverabilityを早期発見するno-go診断」であり、正式Gateを通過させない。
weak/strong効果、w+v0生成、37名集団での安定性、20反復の誤分類率は検証しないため、
作用点を研究結果として主張する根拠には使えない。

実行方法:

```text
cd analysis/pam_dot_task_python
PYTHONPATH=src python3 -B scripts/run_cue_r1_mini.py freeze \
  /Users/utsumikensuke/Research/dot_task/analysis/real_data \
  recovery_runs/cue_r1_mini_0_1_0
PYTHONPATH=src python3 -B scripts/run_cue_r1_mini.py run \
  /Users/utsumikensuke/Research/dot_task/analysis/real_data \
  recovery_runs/cue_r1_mini_0_1_0 --workers 4
PYTHONPATH=src python3 -B scripts/run_cue_r1_mini.py summarize \
  recovery_runs/cue_r1_mini_0_1_0
```

mini screen が `concerning` なら正式R1を止め、どのarchitecture/locusが回収できないかを
調べる。`promising` なら、それは正式R1を実行する価値があるという判断材料にだけ使う。

## 2026-07-24 実行結果

mini screenは4並列で96/96 subject-task、384/384候補fitを完了した。数値失敗は0件、
推定wall timeは1.492時間だった。1 taskの中央値は191.4秒、平均221.9秒だった。

group-level BMSの生成作用点回収は次のとおりだった。

| architecture | 生成セル | 反復1 | 反復2 |
|---|---|---|---|
| parallel | null | null | null |
| parallel | w-medium | null | null |
| parallel | v0-medium | null | null |
| integrated | null | null | v0 |
| integrated | w-medium | w | w |
| integrated | v0-medium | v0 | v0 |

parallelでは、生成真モデルのnullに対する平均log-likelihood改善はw-mediumで0.205、
v0-mediumで0.063にとどまった。追加パラメータの複雑度を含む平均
\(\Delta\mathrm{LME}\) はそれぞれ -1.852、-1.868で、16/16 subject-taskにおいて
生成真モデルがnullのLMEを上回らなかった。これはoptimizer失敗ではなく、現在の
parallel中効果が集団model recoveryには弱すぎることを示す。

integratedでは、nullに対する生成真モデルの平均\(\Delta\mathrm{LME}\) はw-mediumで
+3.780、v0-mediumで+2.322だった。group BMSも両作用点を2/2反復で正しく回収した。
ただしnull生成の1/2反復でv0を選んだため、偽陽性率は小規模診断では不安定である。

結論は `concerning` である。現manifestのまま数週間の正式R1を続けるより、
parallelの効果校正または研究上の候補範囲を事前に再検討し、新manifestを凍結する方が
合理的である。一方、integratedだけに研究質問を狭める場合も、mini screenだけでは
正式Gate通過とはせず、37名・十分な反復による新しい限定Gateが必要である。
