# PAM joint-MAP を37名の dot task へ適用する統合計画

- 作成日: 2026-07-20
- 対象リポジトリ: `antovis86/PAM-PredictiveAccumulationModels`
- 監査時のPAMコミット: `ce8be02` (`main`, `origin/main`)
- 状態: 外部実装中。PAM/TAPAS本体は変更せず、MATLAB参照層と独立Python parity層を管理する。

キュー作用点を行動データから識別する次段階のparallel／integratedモデル、\(w\)／\(v\)／gain候補、回収Gateの実装仕様は [`cue_locus_behavioral_model_spec.md`](cue_locus_behavioral_model_spec.md) に分離して記載する。同仕様は既存 `ddm_w` を上書きせず、案Aのtie規約と本計画の主要likelihood mask・逐次PPC規約を継承する。

## 0. 結論と停止ゲート

### 0.1 実行環境の変更決定（2026-07-20）

研究者が有償MATLABライセンスを利用できないため、37名本解析の実行候補を独立Python再実装へ切り替えた。既存MATLAB層は数式・パラメータ化・短い参照fixtureを作るための凍結参照とし、Python層は `analysis/pam_dot_task_python/` に分離する。Python層ではPAM/TAPASのHGF、WFPT、response model、joint MAP、Ridders微分、BFGS、Hessian、Laplace LMEをソース対応表付きで移植する。

Python推定値は、MATLAB参照fixtureとのHGF軌跡、WFPT、試行別尤度、目的関数、optimizer軌跡、Hessian、LMEの照合、およびparameter recoveryが完了するまで37名の報告値に使用しない。論文・成果物では公式PAMの直接実行ではなく「PAM/TAPASに基づくGPL Python再実装」と明記する。

本解析の推定原則は実現可能である。すなわち、HGF知覚パラメータとDDM応答パラメータを同一の `tapas_fitModel` 目的関数内で、BFGS準ニュートン法により個人ごとにjoint MAP推定し、個人LMEからrandom-effects BMSを行える。

ただし、37名データへの適用には、実装前に研究者が明示的に決めるべき科学的分岐が3つある。

1. **キュー条件付きHGF**
   - TAPAS標準の `tapas_ehgf_binary` は `r.u(:,1)` の単一二値系列を追跡する。
   - 白・赤キューの独立状態を標準設定だけで同時に追跡する機能はない。
   - PAM/TAPASコアを変更せず、外部にカスタムperceptual modelを置き、その内部でキュー別HGFを2本走らせる構成が最小拡張である。この案はjoint MAPを維持できる。
   - **決定（2026-07-20）**: 研究者は第7.2節の案D（外部カスタムperceptual model内でHGFを2本実行）で合意した。実装開始指示に基づき、初期実装はHGFパラメータ共有、両cue共通の中立初期状態、非active cue凍結（cue提示回数軸）で凍結する。

2. **signed coherence のDDMへの導入**
   - 公式 `ddm_hgf.m` は刺激を0/1カテゴリとしてのみ使用し、`ratio` や \(u=2\,ratio-1\) の大きさを使わない。
   - 本課題の難易度情報を反映するにはカスタムresponse modelが必要になる可能性が高い。
   - まず公式DDMを変更なしで参照モデルとして再現し、その後、候補数式とPAMからの逸脱を提示する。既存HGF_DDMの式へ自動的に置換しない。
   - **決定（2026-07-20）**: Gate Bでcoherence対応response modelが承認された。公式driftを入れ子にする \(b_c|\mathtt{signed\_coherence}|\) 項を追加し、\(b_c=0\) で公式モデルに戻る別名response modelとして実装する。

3. **random-effects BMSの版差**
   - 手元のPAM論文PDFは `spm_BMS_gibbs` とexceedance probabilityを指定している。
   - 2026-02-04更新の公式OSF `tutorial.mlx` は `spm_BMS` とprotected exceedance probabilityを使用している。
   - 論文忠実版は `spm_BMS_gibbs` とし、最新版チュートリアル版 `spm_BMS` は感度分析候補とする。最終BMS仕様は37名実行前に固定する。
   - **決定（2026-07-20）**: この方針で合意済み。`spm_BMS_gibbs` を主解析、`spm_BMS` を分離した感度分析とする。

案Dとその初期実装仕様はGate A通過済み、coherence対応response modelはGate B通過済みである。公式 `ddm_hgf.m` は変更せず、入れ子の別名モデルとして実装・検証する。

### 0.2 先行HSSM研究との関係（2026-07-21）

先行研究（ECVP2026ポスター、査読通過済み）は15名 NeuroHaze データのHSSM解析である。**ポスターは変更しない。**

37名版が先行研究の結果を再現しないことが確認されているが、その原因は検出力でも標本でもなく**テスト期の設計差**である。15名版はテスト期を通じて赤cueのみが易しい試行（ratio 0.90）を約13%含むのに対し、37名版はこれを除去し両cue完全均衡としている（第17.2節）。両者は異なる問いに答えており、データセットの差し替えは行わない。

先行研究との関係は**手法比較**として扱う。同一データ・同一モデルにPAMのjoint MAP経路を当て、推定法の違いだけを切り分ける（第17節、Phase 6a/6b）。段階1では静的cue版でHSSM M0–M7を完全に再現し、段階2で信念駆動版へ進む。**正しいベンチマークは試行レベルのLOOではなく被験者レベルのLOSOであり、先行研究のLOSOは全フェーズ有意差なしである。PAMでも帰無が出ることが手法一致の予測である。**

なお、モデル妥当性の主要評価には集約PPCに加えて逐次PPC（時間分解事後予測チェック、第12.3節）を用いる。十分な反復数を含む事後予測バッチを試行index付きで1回生成し、同じ生成配列をウィンドウ別に読み直す。ウィンドウ定義ごとの再生成は行わず、test試行位置別・cue別・coherence難易度別の適合度を同時に評価する。

## 1. 監査した資料と現在地

### 1.1 一次資料

- PAM論文: `/Users/utsumikensuke/Downloads/PAM.pdf`
  - PDF 18–19ページ: joint MAP、BFGS、パラメータ変換
  - PDF 39–42ページ: `u`/`y`、モデル設定、`tapas_fitModel`、集団推論、BMS
- PAM公式コード:
  - `Examples/HGF_examples/DDM_HGF_example.m`
  - `PAM_master/PAM_HGF/DDM/ddm_hgf.m`
  - `PAM_master/PAM_HGF/DDM/ddm_hgf_config.m`
  - `PAM_master/PAM_HGF/DDM/ddm_hgf_transp.m`
- 公式OSFチュートリアル:
  - `tutorial.mlx`, version 3, 更新日 2026-02-04
  - 監査時SHA-256: `94c128d9a041597a775ada43737538a6bd53011c5a7a24343ec26c599f63d26a`
- TAPAS公式ソース:
  - 監査版 6.1.0
  - コミット `7155f99137c5e03f93a2a3afa6a8cb54c75dd4c2`
- SPM:
  - 公式OSFチュートリアルはSPM12を依存関係として指定
  - `spm_BMS_gibbs` の入力は「行=被験者、列=モデル」のLME行列

### 1.2 ローカル環境で未充足の依存関係

監査後、固定コミットのTAPAS 6.1.0とSPM12 r7771を `analysis/pam_dot_task/external/` に配置した。MATLAB実行ファイルは引き続き利用できない。研究者の承認により、有償ライセンスを必要としない独立Python parity実装へ進み、固定ソースとの対応と未通過Gateを `analysis/pam_dot_task_python/PARITY.md` で管理する。

### 1.3 リポジトリの保全状態

- 追跡対象のPAMファイルにローカル差分はない。
- `.DS_Store`、`.gitignore`、`AGENTS.md`、`CLAUDE.md` など既存の未追跡ファイルには触れない。
- 以後、PAM本体の基準ハッシュとGitコミットを各run manifestへ記録する。

## 2. PAM公式コードの推定フロー

### 2.1 入力からjoint MAPまで

```text
被験者CSV
  -> 外部アダプタ（生CSVは不変）
  -> u: trial-wise input
  -> y: [RT秒, choice 0/1]
  -> tapas_fitModel(y, u, perceptual_config, response_config,
                    tapas_quasinewton_optim_config)
       -> perceptual modelが全試行の事前予測を計算
       -> response modelがRT・choiceのtrial-wise log likelihoodを計算
       -> HGF事前 + DDM事前 + 応答log likelihoodから負のlog jointを構成
       -> BFGSでHGF・DDM自由パラメータを同時最適化
       -> native spaceへ逆変換
       -> MAP、軌跡、Hessian、LME、AIC、BICを返す
```

`tapas_fitModel` の目的関数は、知覚・応答パラメータを連結した単一ベクトルに対する負のlog jointである。したがって、HGFを先に固定してからDDMを推定する二段階法ではない。

### 2.2 公式DDMの数式

`ddm_hgf.m` が試行ごとに使う量は次のとおりである。

\[
w_r=0.5+b_w(\hat\mu_r-0.5)
\]

\[
q_r=\operatorname{sigmoid}\left(\frac{1}{\hat\mu_r(1-\hat\mu_r)}-4\right)-0.5
\]

\[
a_r=a_a+b_a q_r
\]

刺激カテゴリを \(s_r\in\{0,1\}\) とすると、

\[
v_r=s_r\{a_v+b_v(\hat\mu_r-0.5)\}
-(1-s_r)\{a_v+b_v((1-\hat\mu_r)-0.5)\}.
\]

したがって、公式モデルでは次が成り立つ。

- `b_w`: 白カテゴリの事前確率がstarting pointを変調する。
- `b_a`: HGFの自由な予測精度パラメータではなく、\(\hat\mu\)から決定論的に作る確信度様の量 \(q_r\) がboundaryを変調する。
- `b_v`: 実際に提示されたカテゴリの事前確率がdrift magnitudeを変調する。
- 刺激強度は0/1カテゴリへ二値化され、coherence magnitudeは入らない。

探索空間上の変換は次のとおりである。

- \(a_a=\exp(\theta_{a})>0\)
- \(a_v=\exp(\theta_{v})>0\)
- \(b_w=2\operatorname{sigmoid}(\theta_w)-1\in(-1,1)\)
- \(b_a,b_v\) は無制約
- \(Ter=\min(RT_{valid})\operatorname{sigmoid}(\theta_{Ter})\)

`a_r` 自体には正値変換がないため、スモークテストでは全試行で \(a_r>0\) を別途検証する。

### 2.3 DDMモデル構成

同一の `ddm_hgf_config` を複製し、使わない傾きのprior varianceを0にして `tapas_align_priors` を呼ぶ。

| モデル | 自由なbelief関連傾き | 0に固定する傾き |
|---|---|---|
| DDM_w | `b_w` | `b_a`, `b_v` |
| DDM_a | `b_a` | `b_w`, `b_v` |
| DDM_v | `b_v` | `b_w`, `b_a` |
| DDM_full | `b_w`, `b_a`, `b_v` | なし |

現行OSFチュートリアルのDDM_v設定には、`m_ddm_v.b_wsa = 0` とすべき箇所で `m_ddm_a.b_wsa = 0` としている疑いの強い行がある。このままではDDM_vが実質的にw+vモデルになる。公式チュートリアルはまず無変更で再現・記録するが、37名用モデルレジストリでは各自由パラメータ集合を自動検査し、モデル名と実体の不一致を許さない。

### 2.4 LME、AIC、BIC

- LMEはMAP点のHessianを用いるLaplace近似である。
- `optim.LME`: 大きいほど良い。
- `optim.AIC`, `optim.BIC`: 小さいほど良い。
- `optim.negLl`: 負の応答log likelihood。
- `optim.negLj`: 負のlog joint。
- `optim.H`, `optim.Sigma`, `optim.Corr`: 局所曲率と事後近似診断に使う。

LME比較は、被験者、尤度対象試行、choice/RT符号、perceptual model定義、事前分布が事前登録されたモデル間でのみ行う。

## 3. 実行層・fixture生成層・再現性記録

本解析は Python 実行層を主系とする。MATLAB/TAPAS は推定の実行依存ではなく、固定した数式・数値経路との短い参照fixtureを生成・更新する層としてのみ保持する。この分離により、MATLAB の有償ライセンスや Online セッションの可用性は37名解析の実行条件ではない。

### 3.1 実行層（37名解析で必須）

`analysis/pam_dot_task_python/` を Python 3.9 以上で実行する。固定する実行依存は `numpy`、`scipy`、`pandas` とプロジェクトの `pyproject.toml` に記載した版範囲である。ここで二キューHGF、DDM尤度、TAPAS互換Ridders/BFGS/Laplace、BMS、parameter recovery、固定3秒のsimulation、aggregate/逐次PPCを実行する。

各 run manifest には、Python version、OS・CPUアーキテクチャ、依存package版、PAM Git commitとtracked diff、Python解析層のcommit/hash、RNG algorithmとseed、perceptual/response/optimizer configの完全なコピー、入力・mask・Gate freezeのhashを保存する。

### 3.2 fixture生成層（任意の参照更新時のみ）

MATLAB、TAPAS/HGF toolbox、PAM本リポジトリ、SPM12は、Python 数値parity fixtureを新しい数式仕様に更新する場合だけ必要とする。MATLAB Online Basic で実行できる最小bundleを用い、private CSVをアップロードしない。Statistics and Machine Learning Toolbox は公式デモの `randsample` やMATLAB版group statisticsにのみ、Parallel Computing Toolbox はMATLAB版の並列化にのみ条件付きで必要である。

fixture生成時は、MATLAB `version` と `ver`、TAPAS/SPM/OSFチュートリアルのversionまたはchecksum、MATLAB pathで解決された `tapas_fitModel`、`tapas_ehgf_binary`、`spm_BMS*`、`ddm_hgf` の絶対パス、bundle hash、RNG algorithmとseedをfixtureのenvironment JSONへ保存する。同名関数のpath衝突は `which -all` で検出し、期待外の実装が先に解決される場合は停止する。

### 3.3 層をまたぐ変更規則

HGF更新、parameter transform、DDM式、尤度対象、deadline、欠測規約、LME定義を変える場合は、Python解析的テストを更新し、必要に応じてfixture生成層で新しい参照を作る。PPCの表示、run manifest、Gateの報告形式など実行層だけの変更は、Pythonテストとparameter recoveryを更新すればよい。旧仕様のfixtureが新仕様の正しさを保証することはない。

## 4. PAMが期待する入力と欠測試行の意味

### 4.1 基本形式

- `u`: \(N\times1\) の二値入力、または追加列を含む \(N\times K\) 行列
- `y`: \(N\times2\) 行列
  - 1列目: RT（秒）
  - 2列目: choice（0/1）

37名用の派生入力候補は次とする。

| `u`列 | 内容 | 使用者 |
|---|---|---|
| 1 | `stimulus_category`（黒=0、白=1） | TAPASのtrial管理、公式DDM、単一HGF参照モデル |
| 2 | `cue_white`（赤=0、白=1） | キュー別カスタムperceptual model |
| 3 | `signed_coherence = 2*ratio_corrected-1` | Gate B承認済みのcoherence対応response model |

`phase`、除外理由、元trial番号などは監査テーブルに保持するが、意図せずモデル共変量にならないよう、当初は `u` へ入れない。

### 4.2 `NaN` の重要な区別

TAPAS 6.1.0の `tapas_fitModel` では、

- `y(r,1)=NaN`: 応答がirregularとなり、その試行の応答尤度だけが除外される。`u`が有限ならHGF更新は続く。
- `u(r,1)=NaN`: 試行全体がignoredとなり、応答尤度から除外されるだけでなく、HGF filteringも停止し、状態表現が一定に保たれる。

本解析では、学習履歴やRT除外試行の刺激をbelief更新へ残したい。そのため、入力 `u` は原則有限のまま保持し、尤度から外す試行は `y` の行全体を `NaN` にする。

## 5. 37名CSVからPAM入力への列対応

### 5.1 被験者・条件

- 対象: `/Users/utsumikensuke/Research/dot_task/analysis/real_data/*_dot_task_*.csv`
- subject ID: 当面はファイルstem全体を一意なIDとして保存する。
- 条件prefixはlongest-prefix-firstで厳密に判定する。
- 未知prefixはエラーにして停止する。

| 条件 | `STIMULUS_REVERSED` | 白判断キー |
|---|---:|---|
| normal | false | j |
| normal_cb | false | f |
| reverse | true | f |
| reverse_cb | true | j |

### 5.2 列変換

| CSV/由来 | PAM監査列 | 規約 |
|---|---|---|
| filename | `subject_id` | stem全体 |
| filename prefix | `condition` | 4条件のみ許可 |
| `main_trial_number` | `trial` | 数値化、1–380、重複・欠番検査 |
| `trial` | `phase` | 1–100=`learning`, 101–380=`test` |
| `rt` | `rt_seconds_raw` | msから秒へ変換 |
| `rt_seconds_raw` | `rt_for_pam` | \(0.15\le RT\le3.0\) かつテスト試行のみ保持。それ以外は `NaN` |
| `response` | `choice_white` | 白判断=1、黒判断=0。未知キーは除外・エラー集計 |
| raw `ratio` | `ratio_raw` | 監査用に不変保存 |
| `ratio_raw`+condition | `ratio_corrected` | reverse系のみ \(1-ratio_{raw}\) |
| `ratio_corrected` | `signed_coherence` | \(2ratio_{corrected}-1\) |
| `ratio_corrected` | `stimulus_category` | \(I(ratio_{corrected}>0.5)\)。**\(ratio=0.5\) は未定義**（第5.2.1節） |
| `ratio_corrected` | `is_tie` | \(I(ratio_{corrected}=0.5)\)。1人あたりテスト40試行＋学習10試行 |
| `cross_color` | `cue` | trim+lower後に `white`/`red` のみ許可 |
| `cue` | `cue_white` | white=1, red=0 |
| 全規則 | `exclude_reason` | `learning_likelihood_mask`, `rt_low`, `rt_high`, `rt_missing`, `choice_missing`, `invalid_key` 等 |

choiceが無効な試行は、RTが範囲内でも `y` の両列を `NaN` にする。TAPASはirregular判定に `y(:,1)` だけを使うため、choiceだけを `NaN` にしてRTを残すことは禁止する。

`rt_for_pam` が `NaN` でも、`rt_seconds_raw` と `choice_white` は監査テーブルに全380行分保持する。学習試行（`learning_likelihood_mask`）の応答は物理的に存在し、応答尤度から除外したconditional held-out・時間外挿データとして逐次PPCで使う（第12.3.2節）。これらの列がPAMの `y` 構築経路へ入らないことをアダプタのテストで検査する。

### 5.2.1 `ratio = 0.5`（tie試行）の扱い

**旧規約の撤回。** 本節は当初 \(ratio_{corrected}=0.5\) を「黒カテゴリ=0」と規約していた。この規約は誤りであり撤回する。

理由は3つある。

1. **客観的な正解が存在しない。** 白100個・黒100個であり、どちらのカテゴリでもない。`dot_task_analysis/docs/spec.md` も同じ理由から、`correct` 列がこの試行では無意味であり正答率分析から除外すべきと明記している。
2. **HGFへ系統的バイアスを注入する。** tie試行を一律「黒を観測した」としてHGFへ入れると、1人あたり50試行（テスト40＋学習10）が黒側へ信念を引く。実際、設計は白3水準・黒3水準で厳密に対称であるにもかかわらず、この規約のせいで \(P(\text{白カテゴリ})\) が 0.500 ではなく 0.429 になる。**この偏りは全て規約由来の人工物であり、実験デザインの非対称ではない。**
3. **最も情報量の高い試行を歪める。** tie試行は事前確率仮説と期待信頼度仮説を分ける診断力が最も高い。ここを規約で決め打ちすることは、検証したい対比そのものを壊す。

**採用できない選択肢。** `u=NaN` によるignored扱いは使えない。`tapas_fitModel` は `irr = unique([ign, irr])` とするため（TAPAS `tapas_fitModel.m:292`）、`u=NaN` は応答尤度からも当該試行を必ず除外する。TAPAS標準では「信念更新はしないが応答尤度には残す」を表現できない。tie試行を捨てると1人280試行のテスト尤度が240試行に減り、かつ最も診断的な試行を失う。

**採用する扱い。** カスタムperceptual model側で、tie試行の第1層更新を恒等写像にする（\(\mu_1 = \hat\mu_1\)、したがって \(\delta_1 = 0\)）。試行はregularのまま残り、応答尤度には寄与する。これは `tapas_ehgf_binary_pu` において \(u=0.5,\ \eta_0=0,\ \eta_1=1\) のとき `und0 = und1` となり自動的に成立する挙動と同一であり、恣意的な特別扱いではなく曖昧刺激に対する正しいベイズ更新である。

**DDM側の規約（2026-07-21 確定・実装済み）。** tie試行の `stimulus_category` が未定義なので、drift の符号項 `direction = 2s-1` も定義できない。**`direction = 0`、したがって `v = 0` とする。**

これは便宜ではなく原理的に正しい値である。白100個・黒100個の刺激はどちらの境界へも証拠を与えないので、drift ゼロが正しい。判断は開始点 `w`（信念由来）と拡散ノイズだけで決まり、tie試行が事前バイアスの純粋なプローブになる。これは本節冒頭で「tie試行の診断力が最も高い」とした性質そのものである。tie以外の試行では公式PAMのdrift式が1ビットも変わらない。

なお本節の初版は「coherence対応モデルなら `direction × b_c × |coh|` が自動的に0になる」と記していたが、**これは誤りである**。`a_v` 項と `b_v` 項が残るため自動的には消えない。撤回する。

`w` は `muhat` のみに依存し刺激カテゴリに依存しないので、tie試行でも通常どおり信念で変調される。境界 `a` も同様である。

**TAPASの `ign` 機構は使えない。** `tapas_ehgf_binary.m` の凍結分岐は `muhat(k,:) = muhat(k-1,:)` とするが、これは1ステップ古い予測である（`muhat(k-1)` は `mu(k-2)` 由来）。tie試行を古い信念で採点することになり、上記の診断的性質を損なう。代わりに、各cueストリームを情報のある試行だけで走らせ、tie試行には「その時点の予測」＝次の情報試行の予測行を割り当てる。末尾tieのためにパディング試行を1つ付け、その予測行のみを読んで更新は捨てる。

この規約は TAPAS に対応物が存在しない我々の拡張であるため、MATLAB parity ではなく**解析的テスト**で検証する（`tests/test_hgf.py::TieTrialTests`、`tests/test_response.py::TieTrialDriftTests`）。既存の `joint.json` fixture は本規約の制定前にexportされたものであり、合成デザインがcue毎に20のtie試行を含むため、当該parityテストは旧規約を明示的に固定している。fixture再生成時に解除する。

### 5.3 正典ローダから継承するものと継承しないもの

`HGF_DDM/src/hgf_ddm/data/dot_task_loader.py` から継承するのは、条件表、ratio反転、キーから判断内容への変換、RT範囲、phase定義である。

同ローダは範囲外RT行をDataFrameから削除するため、その出力をそのままPAMのHGF入力には使わない。行削除すると当該試行の刺激履歴まで失われる。PAMアダプタは380行を保持し、無効な応答だけを `NaN` maskingする。

### 5.4 予備監査結果

これは実装後の正式監査で再計算する。

| 条件 | 人数 | 1人のmain試行 | 学習試行計 | テスト試行計 | RT無効（学習/テスト） |
|---|---:|---:|---:|---:|---:|
| normal | 10 | 380 | 1,000 | 2,800 | 2 / 4 |
| normal_cb | 10 | 380 | 1,000 | 2,800 | 5 / 10 |
| reverse | 7 | 380 | 700 | 1,960 | 2 / 5 |
| reverse_cb | 10 | 380 | 1,000 | 2,800 | 4 / 6 |
| 合計 | 37 | 380 | 3,700 | 10,360 | 13 / 25 |

- 全main試行: 14,060
- RT/choice除外後の主要テスト尤度試行: 10,335
- 各人のcue数: white 210、red 170
- 各人の学習cue数: white 70、red 30
- 各人のテストcue数: white 140、red 140
- 予備監査で有効cue表記は `white` と `red` のみだった。

正式監査では条件別にRT分布、choice比率、coherence分布、刺激×choice整合性、`corr(signed_coherence, choice)`、drift主効果の符号分離の有無も出力する。

## 6. 学習試行とテスト試行の扱い

主要解析は次の仕様とする。

1. `u` は試行1–380をすべて含める。
2. HGFは学習＋テストの全刺激履歴で更新する。
3. 試行1–100は `y=[NaN, NaN]` としてDDM応答尤度から除外する。
4. 試行101–380のうちRT/choice無効試行も `y=[NaN, NaN]` とする。
5. `r.irr` が「学習100試行＋無効テスト試行」と完全一致することをassertする。
6. HGFの \(\hat\mu_r\) は試行rの観測前予測であり、試行r自身の刺激を先読みしていないことを小さな手計算系列でテストする。

この構成はPAM/TAPAS標準の欠測応答機能だけで実現でき、`tapas_fitModel` や `ddm_hgf.m` の変更を必要としない。

## 7. キュー条件付きHGFの監査結果と選択肢

### 7.1 標準実装で分かったこと

1. `tapas_ehgf_binary` は入力第1列の単一系列だけを追跡する。
2. 入力を `NaN` にした試行はmissing observationとして「更新だけを飛ばす」のではなく、予測平均・分散を含む全状態を前試行からコピーして凍結する。
3. `tapas_fitModel` は1つのperceptual functionを呼ぶ。キュー別HGFを2本呼んでactive cueの状態を返す標準configはない。
4. ただし、perceptual function自体は差し替え可能であり、カスタム関数内部で2本を計算しても、パラメータが同じ目的関数へ入っていればjoint MAPは維持される。

### 7.2 4案の比較

| 案 | PAMコア変更 | joint MAP | 2つの独立状態 | 科学的評価 |
|---|---:|---:|---:|---|
| A. 全刺激を単一HGFへ入力 | なし | 維持 | 不可 | cueを無視し、研究目的を表現しない |
| B. cue-stimulus contingencyを単一HGFへ再符号化 | なし | 維持 | 不可 | 1つの共有「contingency belief」になる。本課題の白cueはほぼ中立、赤cueは学習期だけ決定的で対称構造ではないため、2状態の代用にならない |
| C. cueごとに別々にfitして後でDDMへ渡す | PAMコアなし | **喪失** | 可能 | HGFを固定する二段階推定になるため主要解析では不可 |
| **D（採用）**. 外部カスタムperceptual model内でHGFを2本実行 | PAMコアなし | 維持 | 可能 | 最小かつ研究目的に合う。研究者合意済み（2026-07-20）。詳細仕様はGate Aで凍結 |

### 7.3 採用する最小拡張案（案D、アプローチ合意済み）

PAM本体とTAPASを変更せず、外部ディレクトリに `cue_ehgf_binary` 相当のperceptual model/config/transformationを追加する。以下は初期実装としてGate Aで凍結した仕様である。

- 入力: `stimulus_category`, `cue_white`
- 状態: 白cue用HGFと赤cue用HGF
- DDMへ返す量: active cueの観測前 \(\hat\mu\) と対応する予測分散/精度
- HGFパラメータ: cue間で共有
- 初期状態: 両cueとも同じ中立値に固定
- 非active cueの時間意味論: TAPASのignored-trial意味論に合わせ、状態を凍結
- 自由パラメータ: 1セットだけを `tapas_fitModel` のjoint vectorへ入れる

パラメータ共有を採用した理由は、2本の状態は分けつつ「同一参加者の学習特性」は共有でき、赤cue学習試行が30しかない状況での過剰パラメータ化を抑えられるためである。cue別パラメータは小規模検証で識別可能性が確認された場合だけ感度分析候補にする。

非active cueで状態を凍結する仕様は「cueが提示された回数」を時間軸とする。全global trial経過に応じて不確実性を増加させる仕様とは異なる。後者を将来検討する場合は、irregular intervalの扱いを含む別モデルとして明示し、同じモデル名で混在させない。

## 8. signed coherence と公式DDMの境界

### 8.1 公式モデルをそのまま使える範囲

`stimulus_category` を0/1で渡せば、公式 `ddm_hgf` を変更せずfitできる。これはPAMコード経路の再現、choice符号、joint MAP、LMEの動作確認に使う。

### 8.2 そのままでは不足する範囲

本課題のテスト刺激は主に `ratio_corrected=0.35–0.65` であり、RTとchoiceはカテゴリだけでなく0.5からの距離に依存すると予想される。公式DDMは0.51と0.65を同じカテゴリとして扱うため、coherenceを無視するモデルを最終モデルとするにはPPCによる強い根拠が必要である。

公式二値DDMと承認済みcoherence拡張モデルの両方に対し、逐次PPC（第12.3節）で刺激カテゴリ内の \(|\mathtt{signed\_coherence}|\) を難易度帯へビン化し、cueと時間位置を区別した上で観測RT分位点と正答率を評価する。Gate PPCで凍結したglobal discrepancyと同時予測帯、LME、parameter recoveryを組み合わせて、coherence項が必要かを判断する。

### 8.3 承認済み拡張と検証条件

公式PAMの「提示カテゴリに対するbelief」効果を保ち、drift magnitudeに \(b_c|\mathtt{signed\_coherence}|\) を加える。公式 `ddm_hgf.m` を上書きせず、別名のresponse model/config/transformationとして保存する。

37名実行前に次を通す。

1. \(b_c=0\) で公式 `ddm_hgf` の試行別log likelihoodと数値誤差内で一致する。
2. coherence主効果とbelief効果のparameter recoveryと相関を評価する。
3. 元のPAMモデルと拡張モデルのLME、逐次PPC、回収性能を分けて報告する。

## 9. PAM本体の変更要否

### 9.1 変更不要

- CSV読み込みと派生列作成
- RT/choice masking
- 学習試行をHGFへ残しDDM尤度から外す処理
- DDM_w/a/v/fullのprior variance設定
- subject/model run wrapper
- 結果保存、監査、再開、BMS

### 9.2 PAM/TAPASコアを変更せず外部拡張が必要

- キュー別HGFを同一joint objectiveで計算するperceptual model
- coherence magnitudeを使う場合のresponse model

### 9.3 変更禁止境界

- `PAM_master/PAM_HGF/DDM/ddm_hgf.m` の直接編集
- `tapas_fitModel` やTAPAS optimizerの編集
- HGFを外部で先にfitし、そのMAPを固定してDDMだけfitする変更
- HSSM/PyMC/NumPyroへの置換
- 生CSVの変更

コア編集が不可避と判明した場合は、理由、対象行、代替案、joint MAP/LMEへの影響を報告して停止する。

## 10. 公式デモの再現手順

依存関係を揃えた後、37名コードより先に次を行う。

1. PAMコミットとtracked hashを保存する。
2. TAPAS/SPM/MATLABのversionと解決pathを保存する。
3. `Examples/HGF_examples/DDM_HGF_example.m` をファイル変更なしで実行する。
4. RNG seedを呼出し側で固定し、seedとalgorithmを記録する。
5. 出力 `m` に以下があることを確認する。
   - `p_prc`, `p_obs`
   - `traj.muhat`, `traj.sahat`
   - `optim.negLj`, `optim.negLl`
   - `optim.LME`, `optim.AIC`, `optim.BIC`
   - `optim.H`, `optim.Sigma`, `optim.Corr`
6. 同seedで結果が再現することを確認する。
7. 実行後もPAM tracked hashが不変であることを確認する。
8. 公式OSFチュートリアルは別runとして無変更実行し、現行コードとの不整合やエラーを記録する。

公式デモはシミュレーションに `randsample` を使い、スクリプト内ではseedを固定していない。再現性は外側のrunnerで担保し、公式ファイルは編集しない。

## 11. 1名スモークの具体的手順

### 11.1 代表被験者

予備監査で中央値に近かった次を第一候補とする。

`normal_cb_dot_task_20260526_013329_6717f0ac88d9f27d9b79af31`

- テスト有効試行: 280/280
- テストRT中央値: 0.871秒
- 白判断比率: 0.500

小規模検証では必ずreverse系被験者も含め、ratio反転経路を通す。

### 11.2 スモーク順序

1. 380行の監査テーブルを作る。
2. 条件、trial、cue、ratio、choice、RT、maskのassertを通す。
3. `u` は380行すべて保持する。
4. `y` は学習100行を `NaN`、テスト280行を有限値とする。
5. 最初は公式単一HGF＋公式DDM_wをfitし、PAM経路だけを確認する。
6. DDM_a、DDM_v、DDM_fullへ広げる。
7. Gate Aで仕様を凍結したキュー別HGFを同じ被験者でfitする。
8. Gate Bで承認されたcoherence対応モデルを同じ被験者でfitする。

### 11.3 合格基準

- BFGSが有限の解を返す。
- `optim.negLj`, `optim.negLl`, LME, AIC, BICが有限。
- HGF自由パラメータとDDM自由パラメータの両方が初期値から更新される。
- 全有効試行で \(0<w<1\), \(a>0\), \(Ter<RT\)。
- belief、予測分散/精度、drift、trial log likelihoodが有限。
- `muhat` が観測前予測である。
- 学習試行の応答log likelihoodが除外され、HGF更新は残る。
- 同一入力・同一seed・同一configで結果が再現する。
- Hessian/Sigmaの数値状態とパラメータ相関に致命的な異常がない。
- posterior predictive simulationでchoice率、RT中央値、RT分位点が明らかに破綻しない。
- 試行indexを保持した逐次PPC（第12.3節）で、test試行位置別・cue別のchoice率とRT分位点に明らかな時間依存の破綻がない。

### 11.4 複数初期値

TAPAS 6.1.0は `nRandInit>0` を持つが、固定 `seedRandInit` を各反復内で再設定するため、同じ呼出し内で複数の同一乱数初期値を作る可能性がある。TAPASを編集せず、外部runnerから異なるseedで独立に `tapas_fitModel` を呼び、各runを保存する。

- prior mean開始を1回
- 事前分布からの決定論的な異なる開始点を最低8回
- 最良の有限LMEだけでなく、各runの `negLj`, LME, MAP、終了理由を保存
- 実質的に異なる局所解がある場合は3–5名への拡張を停止

## 12. 小規模検証とparameter recovery

### 12.1 3–5名

最低4条件から各1名を含める。5人目を入れる場合はRT/choice除外数が最大または分布端に近い被験者を選ぶ。

確認するもの:

- 最適化成功率と実行時間
- 初期値感度
- HGF MAPとbelief軌跡
- DDM MAPとtrial-wise \(w,a,v\)
- LME/AIC/BIC
- 予測と観測のchoice率、RT q10/q50/q90
- cue別、coherence別、test trial位置別の逐次PPC（第12.3節）
- conditionによるdrift主効果の不自然な完全分離がないこと

### 12.2 回収検証

公式PAMのシミュレーション手順を基準にし、少なくとも次を段階的に行う。

1. 公式単一HGF＋公式DDMの既知パラメータ回収
2. 37名の実デザイン行列を使ったキュー別HGF＋公式DDMの回収
3. coherence拡張を採用する場合、そのモデル固有の回収
4. null effect、wのみ、aのみ、vのみ、fullを区別できるか確認
5. HGFパラメータとDDM傾きの相関・トレードオフを確認

**案A確定後のV3 Gate（結果確認前に固定）。** tieをHGFでは更新なし、DDMでは `direction=0`・\(v=0\) とする最終仕様に対し、従来の `ddm_v` Grid V2だけでは中心仮説の開始点効果 `b_w` を検査できない。このため、V2の不通過後に閾値を変更するのではなく、科学的役割が異なる次の2 gridを新規versionとして固定し、**両方の全自由パラメータが既存 `RECOVERY_CRITERIA_V1` を通過すること**を37名fitの必要条件とする。

1. `recovery-grid-ddm_w-tie_v0-3.1.0`: 32ケース。案Aの中心仮説である `omega_2`–`b_w` 経路を縮約モデルで検査する。
2. `recovery-grid-ddm_full_c-tie_v0-3.1.0`: 32ケース。`b_w`, `b_a`, `b_v`, `b_c` を同時に含む最大保持候補を検査する。

各gridはorder-32 Walsh contrastでtruth列を厳密に無相関化する。\(\omega_2\) はBayes-optimal範囲を覆う \(-5.5,-4.9,-3.7,-3.1\) の4水準、その他はsupport内のbalanced 2水準とする。seed、digest、criteriaは `manifests/recovery_gate_v3_freeze.json` に保存する。相関基準0.70、絶対bias 0.50、prior SD比RMSE 1.0、各パラメータ20ケース以上という基準はV2から変更しない。

V3.0の最初の `ddm_w` 生成データは、回収推定に入る前のprior-mean初期値でWFPT尤度が数値support外となった。この実行可能性失敗を保存した上で、`log_a_a` truthだけを `{0.4,0.8}` から実データMAP周辺の `{0.2,0.6}` へ狭め、V3.1へversionを上げた。V3.1の全64 seedについて生成データの初期目的関数が有限であることだけを事前検査し、回収推定値・相関・bias・RMSEを一切見ずに上記digestを固定した。

同じデータを何度もfitする最適化再現性と、異なる合成データセットを生成する統計的回収を分けて報告する。

### 12.3 逐次PPC（時間分解事後予測チェック）

全テスト試行をまとめたchoice率・RT分位点だけの集約PPCは、学習に伴うbeliefの時間変化を平均で潰す。本課題ではbeliefが学習期からテスト期にかけて動き、activeなcueも試行ごとに切り替わるため、集約統計が一致していても時間依存の系統的なmisfit（学習初期だけ予測が速すぎる、赤cueの学習後半だけchoice率がずれる等）は検出できない。そこで主要な妥当性評価に逐次（時間分解）PPCを加える。

#### 12.3.1 基本原理

「1回生成」とは1本の確率的応答系列を指すのではなく、事前に固定した十分な反復数 \(S\) を含む1つの生成バッチを指す。時間分解のためにウィンドウごとの再生成は行わない。

1. 事後予測反復数 \(S\) と生成方式（Laplace事後draw、またはMAP固定の応答反復）を事前に固定する。
2. 観測と同一の `u` 系列・同一の試行順・同一のactive cueのまま、各反復で全試行のRTとchoiceを生成する。Laplace事後が妥当な場合は反復ごとにパラメータをdrawし、使用できない場合は同一MAPから独立な応答系列を \(S\) 本生成する。
3. 生成結果を集約せず、試行indexを保持した \(S\times N\times2\)（反復×試行×{RT, choice}）配列として保存する。各反復にparameter draw ID、応答生成seed、生成方式を付与する。
4. 保存した配列を時間位置を保ったままウィンドウへ射影し、ウィンドウごとに観測と予測の統計量を対にして比較する。

すなわち「十分な反復を含む一度の事後予測生成バッチでも、時間位置を保ったままウィンドウ別の統計量を計算すれば、時間分解PPCになる」。ウィンドウ定義を変えるたびに再生成する必要はなく、同一の生成バッチを異なる分割で読み直す。

#### 12.3.2 ウィンドウ定義

同一の生成配列に対し、以下の分割を並行して適用する。

| ウィンドウ | 定義 | 検出したい破綻 |
|---|---|---|
| test試行位置 | 試行101–380を事前固定したブロックまたはスライディング窓で分割 | テスト期を通じた学習/ドリフトの再現失敗 |
| cue別×位置 | 白cue系列・赤cue系列それぞれの提示順index | cue別HGFの状態分離が実データと合わない |
| coherence難易度 | 刺激カテゴリ内で \(|\mathtt{signed\_coherence}|\) をビン化し、cueと時間位置を区別 | 公式二値DDMがcoherenceの大きさを無視することの影響（第8.2節） |
| 学習期（conditional held-out） | 試行1–100を事前固定したブロックで分割 | 学習期のbelief軌跡への時間外挿が非現実的 |

cue別ウィンドウはGate Aで凍結したcue提示回数軸で切る。global trial軸と混同しないよう、両軸の図を別々に出す。

学習期ウィンドウは追加の診断価値がある。学習100試行のRT・choiceは生CSVに存在し（第5.2節の `learning_likelihood_mask` は尤度から外すだけで、データを消すわけではない）、モデルはこれらの応答を尤度計算に使わない。ただし、パラメータは同一参加者の後続テスト応答から推定される。したがってこれは独立データによる前向き予測ではなく、「応答尤度から除外した学習期へのconditional held-out・時間外挿チェック」と位置づける。予測と観測が合っても外部検証より強い証拠とは解釈しない。学習期だけ乖離する場合は、belief軌跡の初期挙動かTerを含むDDMの定常性仮定を疑う。

この用途のため、学習試行の `rt_seconds_raw` と `choice_white` は、`y` から除外した後もPPC比較用の監査列として保持する（第5.2節の派生列に含める）。尤度計算に使わないことをコード上で明示し、誤って `y` へ戻らないよう検査する。

#### 12.3.3 統計量と事後不確実性

各ウィンドウ×観測/予測で比較する量:

- choice率（白判断率）
- 正答率（特にcoherence難易度別）
- RT分位点 q10/q50/q90
- choice-conditional RT（白判断時・黒判断時のRT中央値）

事後予測分布は、第12.3.1節の \(S\) 反復でパラメータ不確実性と応答生成ノイズを表現する。MAP周辺のLaplace事後を使える場合は、各反復で `optim.Sigma` からパラメータをdrawし、対応するHGF軌跡を再計算して応答を1系列生成する。Laplace事後を使えない場合は、MAPを固定した複数の応答生成により、少なくとも応答ノイズの予測分布を得る。後者はパラメータ不確実性を含まないことを図表と診断に明記する。

- 生成反復数 \(S\) は事前に固定し、run manifestへ記録する。
- 生成に使う乱数のseedとalgorithmを記録し、再現を確認する。
- TAPASが保存する `optim.Sigma` は自由パラメータの変換空間（BFGSが最適化した無制約空間）上にある。自由成分をdrawした後に固定パラメータを含む完全ベクトルを復元し、`*_transp` を通してnative spaceへ写す。
- TAPASは数値Hessianが非正定値の場合にBFGS逆Hessianへ退避し、`nearest_psd` 補正を行うことがある。外部診断で補正前の数値Hessianの有限性・最小固有値・条件数と退避経路を記録し、出力 `optim.Sigma` が正定値に見えることだけを採用条件にしない。
- 補正前の数値Hessianが非正定値、条件数が事前閾値を超える、またはSigmaの次元が自由パラメータ数と一致しない場合はLaplace drawを使わず、MAP固定の \(S\) 応答反復に退避する。
- 各drawは変換後に、有限なHGF軌跡、有限な目的関数、および全試行のDDM support（例: \(0<w_r<1\), \(a_r>0\), \(Ter>0\)）を検査する。不正drawはclipせず棄却・再生成し、試行数 \(S\) を満たすまでの提案数、棄却数、棄却率を保存する。棄却率が事前閾値を超えた場合はLaplace近似不良とする。

各ウィンドウの点ごとのPPC区間と予測percentileは可視化用の記述診断とする。予測percentileから導く両側tail probabilityは計算方法を固定して報告するが、個々のウィンドウを独立な有意性検定として扱わない。ウィンドウ数と統計量数による偶然の外れを制御するため、事前宣言した全ウィンドウ×主要統計量に対する最大標準化偏差をglobal discrepancyとして各反復で計算し、その予測分布から同時予測帯を作る。標準化法、tail定義、同時予測帯の水準、および公式モデルとGate B承認済み拡張モデルの採否に用いる「系統的乖離」の定量基準は、Phase 3のPPC結果を見る前にGate PPCで凍結する。

#### 12.3.4 実装上の必須条件

- 生成時も \(\hat\mu_r\) は試行rの観測前予測でなければならない。逐次PPCは因果順序が崩れると意味を失うため、第6節6項および第11.3節のmuhat検査を逐次PPCの前提条件として扱う。
- 尤度から外した試行でも刺激履歴はbeliefへ入るため、全380試行について予測を生成する。ただし比較の位置づけを厳密に分ける。学習100試行はconditional held-out・時間外挿チェック（第12.3.2節）、テスト有効試行はin-sample適合チェックであり、両者を同じ表に混ぜない。RT/choiceが物理的に無効な試行（`rt_low`, `rt_high`, `choice_missing` 等）は観測値が存在しないため、どちらの比較からも除外し、ウィンドウ別に除外数を明示する。
- RT予測の生成にはDDMの第一通過時間シミュレータが必要である。`ddm_hgf.m` は尤度評価のみを行い生成関数を持たない可能性があるため、Phase 1でPAM/TAPASが提供する生成経路（`tapas_simModel` 系）の有無を確認する。存在しない場合は、`ddm_hgf.m` を書き換えず外部シミュレータを別名で追加し、そのパラメータ化が尤度側と厳密に一致することを、既知パラメータでの尤度対シミュレーション整合テストで検証する。
- 観測が疎なウィンドウ（赤cueの学習30試行など）は統計量が不安定になる。ウィンドウあたりの有効試行数を必ず併記し、少数ウィンドウの乖離を過剰解釈しない。
- ウィンドウ幅と分割境界は結果を見る前に固定する。事後にウィンドウを選び直して極端な乖離を探すことは禁止する。

#### 12.3.5 使用箇所

逐次PPCは以下で参照する。

- 第11.3節: 1名スモークの合格基準（明らかな時間依存の破綻がないこと）
- 第12.1節: 3–5名検証の主要な適合度評価
- 第8.2節: coherenceの大きさを無視する公式二値DDMを最終モデルとしてよいかの判断材料。刺激カテゴリ内のcoherence難易度ウィンドウで、事前固定したglobal discrepancyと同時予測帯に基づく系統的乖離が出る場合、Gate B承認済み拡張モデルを最終モデル群に残す根拠の1つとする
- 第13.2節: 37名実行での被験者×モデル単位の保存対象

## 13. 37名本解析の実行設計

### 13.1 モデルレジストリ

37名実行前に、各モデルについて次を凍結する。

- model IDと表示名
- perceptual modelとHGF階層数
- cue間で共有/分離するパラメータ
- response modelの完全な数式
- 自由/固定パラメータ一覧
- native-space変換
- prior mean/variance
- optimizer設定
- likelihood対象試行
- code hash

PAM論文チュートリアルに合わせたeffective 2-level HGFを最初のスモーク対象とする。3-level HGFを本解析候補にする場合は、小規模検証で安定性を確認し、2-levelとの関係を事前に定義する。

### 13.2 出力構造

PAM本体とは別の解析ディレクトリに、run ID単位で保存する。

```text
run_manifest
input_audit
model_registry
subjects/
  <subject_id>/
    <model_id>/
      fit.mat
      summary.json
      trialwise.csv
      diagnostics.json
      log.txt
bms/
group/
```

各被験者×モデルについて保存する。

- HGF MAP（native/transformed）
- DDM MAP（native/transformed）
- trial-wise active-cue prior belief
- HGF予測分散/精度とPAM boundary modulatorを区別した列
- trial-wise \(w,a,v,Ter\)
- `Ter` 診断：尤度に含めた最小RT、`Ter/min(RT_valid)`、最小RTまでの残余決定時間、および変換空間の `Ter_logit`。これは境界近傍を可視化する診断であり、結果を見て閾値を追加しない。
- `negLj`, `negLl`, LME, AIC, BIC
- Hessian、Sigma、主要パラメータ相関
- 初期値、seed、optimizer履歴、終了理由
- irregular/ignored trial index
- 逐次PPC用の試行index付き事後予測バッチ（反復×試行×{RT, choice}、parameter draw ID、生成方式を含む）
- 逐次PPCのウィンドウ別観測/予測統計、有効試行数、点ごとのPPC区間、global discrepancy、同時予測帯、生成seed
- Laplace drawの使用可否、補正前Hessian診断、退避経路、不正drawの棄却数・棄却率
- 使用した入力とconfigのhash
- 実行時間
- 成功/失敗理由

### 13.3 失敗復旧

- 1被験者×1モデルを再開単位にする。
- 完成済みでhashが一致する結果は再実行しない。
- 一時ファイルへ保存後、検証を通ったものだけ完成名へ移す。
- 失敗は握りつぶさず、例外、seed、開始点、最終有限値を保存する。
- 自動retryは事前定義した異なる初期値だけに限定する。
- model定義やpriorを変えたretryは新しいmodel/run IDにする。
- 37名BMSでは、全比較モデルが成功した被験者の共通集合だけを使い、LMEを補完しない。

### 13.4 集団解析

- 個人MAP値の記述統計と分布を報告する。
- belief関連傾きについて、論文に合わせた両側one-sample testを行う。
- belief関連傾きについて「実質的に効果がない」と結論する場合は、両側検定の非有意性ではなく one-sample TOST（\(\alpha=0.05\)）を用いる。個人MAP値の群平均を群標準偏差で割った標準化効果量 \(d=\bar\theta/s_\theta\) に対し、等価限界を事前に \([-0.20,+0.20]\) と固定する。効果量の90%信頼区間全体がこの範囲に含まれる場合にのみゼロと実質的に等価と判定する。これは「小さい効果」の上限を Cohen の \(|d|=0.20\) とする宣言であり、結果を見て変更しない。
- PAMと既存HSSMの群レベル結論を比較するときも、それぞれの方法で得た標準化効果量と90%区間を同じ \([-0.20,+0.20]\) の基準で分類し、`positive`、`negative`、`equivalent-to-zero`、`inconclusive` の4区分を並べる。異なるデータセット・階層構造の点推定値を同一母集団のpaired差として直接TOSTしない。
- 条件差を検討する場合、4条件のカウンターバランス要因と科学的条件を混同しない。
- 効果量、信頼区間、多重比較方針を事前に固定する。
- MAP対既存MCMCの比較は別解析として明示し、PAM結果と混同しない。

## 14. Random-effects BMS計画

### 14.1 論文忠実版

`spm_BMS_gibbs(LME, alpha0, Nsamp)` を使用する。

- LME行列: 行=被験者、列=モデル
- `alpha0=ones(1,n_models)` を明示
- `Nsamp` を明示
- RNG seedを直前で固定
- `exp_r`, `xp`, `r_samp`, `g_post` を保存
- DDM_w/a/v/full内で比較

Gibbs関数は乱数を使うため、同seed再現と、seedを変えたときのxpのMonte Carlo誤差を確認する。

### 14.2 最新OSF感度版

現行 `tutorial.mlx` と同じ `spm_BMS` を使い、`exp_r`, `xp`, `pxp`, `bor` を保存する。論文忠実版と同じ表に混ぜず、手法名を明示する。

### 14.3 比較順序

本研究の主要対象がDDMである限り、まずDDM構成内のBMSを行う。RDM/LNRまで拡張する場合のみ、論文と同じくfamily内勝者を決め、その後family勝者間を比較する。

## 15. 実施フェーズと承認ゲート

### Phase 0: 依存関係

- MATLAB、TAPAS、SPM12を導入・固定
- version/path監査
- **合格後のみPhase 1へ**

### Phase 1: 公式再現

- リポジトリDDM demoを無変更実行
- OSF tutorialを別runで無変更実行
- tutorialのモデル名/自由パラメータ不一致を記録
- BFGS、MAP、LME/AIC/BIC、BMSの経路確認
- **合格後のみPhase 2へ**

### Phase 2: 37名入力アダプタ

- 生CSVを読み取り専用で変換
- 37/37監査
- 380行保持とmask意味論のテスト
- **合格後のみPhase 3へ**

### Gate A: キューHGF仕様の凍結（通過済み）

アプローチは案D（外部カスタムperceptual modelで2本のHGFを実行）で合意済み。初期実装は以下の仕様で凍結した。

- cue間のHGFパラメータは1セットを共有する
- 非active cueは状態を凍結し、cue提示回数を時間軸とする
- 両cueは同じ中立初期状態から開始する
- DDMへactive cueの観測前 \(\hat\mu\) と予測分散/精度を返す

### Gate PPC: 逐次PPC仕様の事前凍結

Phase 3の1名スモークでPPC結果を見る前に、以下をrun manifestで凍結する。これにより、ウィンドウや判定基準を結果に合わせて選び直すことを防ぐ。

- ウィンドウ定義と分割境界
- 生成反復数 \(S\)、seed、乱数algorithm、decision-time格子（実験の反応期限3秒で固定）
- Laplace事後drawの採用条件、Hessian条件数・draw棄却率の閾値、MAP固定生成への退避規則
- 主要統計量、点ごとのtail probability定義、global discrepancyの標準化法、同時予測帯の水準
- 公式モデルとGate B承認済み拡張モデルの採否に用いる「系統的乖離」の定量基準

### Phase 3: 1名スモーク

- 公式モデル
- 仕様凍結後のキュー別モデル
- Gate B承認済みのcoherence拡張モデル
- 複数初期値、集約PPC＋逐次PPC（第12.3節）、数値診断

### Gate B: coherence数式の承認（通過済み）

- 採用式: \(v_r=d_r[a_v+b_c|c_r|+b_v(p_{\mathrm{presented},r}-0.5)]\)
- \(d_r=2s_r-1\), \(c_r=\mathtt{signed\_coherence}_r\)
- \(b_c=0\) で公式二値DDMのdriftと厳密に一致させる
- 公式 `ddm_hgf.m` を上書きせず、別名のconfig/transformation/response modelで管理する
- 公式モデルへの入れ子テストとparameter recoveryを37名実行前に通す

### Phase 4: 3–5名＋回収

- 全条件を含む小規模検証
- parameter recovery
- 初期値感度
- **合格後のみPhase 5へ**

### Gate C: 37名モデルレジストリの凍結

- HGF階層数
- cueパラメータ共有
- DDM数式
- prior
- BMS方式
- seed/optimizer設定
- Gate PPCで事前凍結したウィンドウ、生成反復数 \(S\)、seed、Laplace/MAP退避規則を再確認する
- Gate PPCで事前凍結したtail probability、global discrepancy、同時予測帯、系統的乖離の定量基準を、結果依存で変更していないことを確認する

### Phase 5: 37名

- 被験者×モデル単位で実行・再開
- 個人MAP、LME、逐次PPC
- random-effects BMS
- 個人MAP値の集団解析

### Phase 6a: HSSM手法比較・段階1（静的cue版、15名 NeuroHaze）

第17節の仕様に従う。37名版の主解析とは独立の系統であり、Phase 5 と並行して進められる。

- `ddm_hgf_linear` を別名で実装（第17.4節）。`ddm_hgf.m` は変更しない
- HSSM M0–M7 を偏差項の自由/固定で構成（第17.5節の対応表）
- 前処理をHSSMと一致させる（第17.6節）。尤度対象試行を1試行単位で照合
- 15名×3フェーズ×8モデルを推定し、フェーズ別random-effects BMS
- M4の \(d_{v0}\)–\(d_w\) 相関を先行研究の \(\hat R\) 除外判断と突き合わせる
- **ベンチマークはLOOではなくLOSO**（第17.7節3項）。帰無一致が成功である
- **合格後のみ Phase 6b へ**

### Phase 6b: HSSM手法比較・段階2（信念駆動版）

- 案D + cue別HGF を15名データへ適用
- 静的cue版と信念駆動版をBMSで直接比較
- 15名版テスト期に残る赤cueのcontingency（第17.2節）を信念がどう表現するかを逐次PPCで確認
- HSSMのモデル空間に無い境界変調（`b_a`）をここで初めて検証する

### Python parity実装状況（2026-07-20）

独立Python層では、入力アダプタ、案Dの2-cue HGF、公式およびcoherence拡張DDM、WFPT、joint MAP、TAPAS式Ridders/BFGS/Hessian/Laplace LMEに加え、次を実装済みである。

- 論文忠実版 `spm_BMS_gibbs` と、感度分析用protected exceedance BMS
- PAM DDMからの試行index付きMAP固定予測生成
- Laplace事後パラメータdraw、不正draw棄却、診断付きMAP固定退避
- 同じ予測生成バッチを再生成せずに読む集約PPC（7ウィンドウ）と逐次PPC（49ウィンドウ）
- 点ごとの予測区間・tail probability、最大標準化偏差によるglobal discrepancy、同時予測帯
- 元の試行設計と尤度maskを保つparameter recovery生成・再推定runner（生成後の最小RTに合わせた真の `Ter_logit` 補正を含む）

37名の読み取り専用監査では全員について49ウィンドウを構成でき、指定1名の100反復×380試行の非報告用スモークでは、同じ生成配列を集約・逐次PPCの双方が変更せず再利用した。これは配線とshapeの検査であり、初期パラメータ・100反復から得たPPC値は解釈しない。

未通過なのは、MATLAB参照fixtureによるHGF/WFPT/尤度/optimizer/LME/BMS/PPCの数値照合、Gate PPCにおけるHessian条件数・draw棄却率の閾値凍結、事前宣言した十分なgridでのparameter recovery合格である。3秒格子のcaptured massは期限内反応確率として保存し、RT・choice PPCは有効な期限内反応に条件づける。Laplace drawと退避経路、およびrecovery runnerの実装テスト通過だけを科学的Gateの通過とはみなさない。これらが完了するまでPython推定値・PPC・BMSを37名の報告値に使用しない。

2026-07-21にMATLAB Online R2026a Update 4でexportした決定論的fixtureについて、`design.json` はPython再構成入力と完全一致した。`ddm.json` の公式5ケースおよびcoherence拡張6ケースでは、全試行対数尤度の最大絶対差は \(3.6\times10^{-15}\)、各ケースの総和差は最大 \(1.2\times10^{-13}\) であり、DDM response/WFPT尤度の移植は浮動小数点丸めの範囲で一致した。この照合は継続テストとして保存したが、joint objective、HGF、BMS、PPCのGateは別途未通過である。

同じMATLAB Online出力の `hgf.json` では、3つの \(\omega_2\) 値についてsingle-stream eHGFと案Dの2-cue eHGFの \(\mathtt{muhat}\)、\(\mathtt{sahat}\)、\(\mathtt{mu}\)、\(\mathtt{sa}\) 全軌跡がPythonと最大 \(2.2\times10^{-16}\) で一致した。`wfpt.json` の2,240点も最大 \(1.1\times10^{-15}\) で一致した。従って入力アダプタを除くHGF・WFPT・DDM trial likelihoodの数値parity Gateは通過済みとし、joint MAP/LME、BMS、simulation/PPC、およびparameter recoveryを残す。

`bms.json` ではprotected BMSの \(\alpha\)、期待model frequency、BORがMATLABと最大 \(2.4\times10^{-15}\) で一致した。Gibbs BMSおよびDirichlet exceedance probabilityはMATLABとNumPyのGamma乱数実装がdraw-for-drawでは一致しないため、同一seedの独立chainとして比較し、expected frequency・exceedance probability・平均subject posteriorの最大差 \(1.3\times10^{-2}\) を事前宣言した \(0.02\) のMonte Carlo許容幅内と確認した。BMS parity Gateも通過済みとし、joint MAP/LME、simulation/PPC、およびparameter recoveryを残す。

`simulation.json` では、3秒で固定したWFPT格子のtrial-wise captured massが最大 \(1.3\times10^{-15}\) で一致した。MATLABとNumPyは反応生成の乱数列を共有しないため、200反復×380試行のRT平均・RT中央値・choice率はtrial-wise RMSE \(\leq 0.06\) の分布的照合とし、実測最大RMSEは \(0.040\) 未満だった。DDM simulation kernelのparity Gateは通過済みである。続く `ppc.json` では、同じ200×380生成バッチをaggregate 7 windowと逐次49 windowで再利用した。window ID・family・有効試行数・観測統計量は最大差 \(1.2\times10^{-16}\) で一致し、独立乱数列による予測要約も事前宣言したRMSE許容幅内だった。aggregate/逐次PPC summaryのMATLAB fixture照合も通過済みとする。

`joint.json` では、MATLAB Online が得た MAP 変換空間パラメータを Python の同じ joint objective に入力し、負の対数尤度・負の対数事後がともに最大差 \(5.7\times10^{-14}\) で一致した。MATLAB がexportした Hessian は正定値で、exportした共分散との逆行列関係を満たし、その Laplace 式からLMEも再現した。さらに、Python は宣言済みの事前平均から独立に局所解を回復し、変換空間パラメータ差は最大 \(0.02\)、負のlog joint差は \(10^{-3}\) 未満だった。その点をTAPAS互換BFGSで局所確認してRidders Hessianを計算し、正定値のLaplace近似とMATLAB LMEとの差 \(0.03\) 以内を確認した。逐次PPC summaryのMATLAB fixture照合も通過済みである。

案Aを最終仕様（tieではHGF更新を保持し、DDMは `direction=0`、従って \(v=0\)）として固定後、`RECOVERY_GRID_V2` の24ケースを再実行した。24/24がgradient toleranceでresetなしに収束したが、凍結済みGateは不通過だった。`log_a_a`、`log_a_v`、`Ter_logit` は全基準を通過した一方、`hgf.omega_2` の回収相関は \(0.092\)、`ddm.b_v` は \(0.674\) で、事前固定した \(0.70\) を下回った。閾値は結果後に緩和しない。`Ter/min(RT_{valid})` は \(0.776\)–\(0.927\)、最速反応に残る決定時間は \(0.033\)–\(0.112\) 秒だった。この結果は失敗Gateとして保存し、原因を解決して新たに事前固定した回収設計を通過するまで37名本fitへ進まない。

続いて、結果確認前に固定したV3.1を実行した。`ddm_w` 32ケースは32/32がresetなしに収束し、`b_w` は相関 \(0.950\) でbias・RMSEを含む全基準を通過したが、`omega_2` は相関 \(0.547\) のため不通過だった。`ddm_full_c` 32ケースも32/32が収束し、8/8パラメータが全基準を通過した（`omega_2` \(0.815\)、`b_w` \(0.902\)、`b_a` \(0.883\)、`b_v` \(0.781\)、`b_c` \(0.711\)）。ただしV3.1の事前規則は両gridの合格を要求したため、複合Gateは不通過とする。fullモデルの成功を見て規則を変更しない。V3.1全体の `Ter/min(RT_{valid})` は \(0.820\)–\(0.985\)、最小残余決定時間は5.8 msであり、小規模実データ検証で境界近傍診断を必須とする。

## 16. 最終的な成功条件

この統合を成功とみなす条件は次である。

1. PAM/TAPASのBFGS joint MAP経路が維持されている。
2. 生CSVとPAM本体が変更されていない。
3. 学習履歴はbeliefへ入り、主要DDM尤度はテスト試行だけである。
4. choice、ratio反転、cue、RTの符号規約が37/37名で監査済みである。
5. キュー別beliefが観測前予測であり、active cueだけがDDMを駆動する。
6. HGFとDDMの両パラメータが個人ごとに同時推定される。
7. 局所解、無効boundary、非有限尤度、Hessian問題が可視化される。
8. 拡張モデルは公式PAMモデルと別名・別hashで管理される。
9. 37名実行前にparameter recoveryと3–5名検証を通過する。
10. LMEとrandom-effects BMSの版・seed・入力行列が再現可能である。
11. 十分な反復を含む逐次PPC（時間分解PPC）で、test試行位置別・cue別・cue・時間を区別したcoherence難易度別のchoice率・正答率・RT分位点を評価し、事前固定したglobal discrepancyと同時予測帯に基づく系統的乖離をモデル選択の議論へ明示的に反映している。
12. belief関連効果について「効果なし」またはHSSMと同じく実質的にゼロと主張する場合、標準化効果量の90%信頼区間が事前固定した等価限界 \([-0.20,+0.20]\) に完全に含まれている。通常の両側検定が非有意であることだけでは、この条件を満たさない。区間がゼロを含むが等価限界内に収まらない場合は `inconclusive` と報告する。

## 17. HSSM先行研究との手法比較（15名 NeuroHaze データ）

### 17.1 目的

先行研究（ECVP2026ポスター、査読通過済み、**変更しない**）は15名の NeuroHaze データに階層ベイズHSSM 8モデル（M0–M7）をフェーズ別に当て、PSIS-LOO/LOSOで比較した。本節は、同一データ・同一モデルにPAMのjoint MAP経路を適用し、**推定法の違いだけを切り分けて**両者が同じ結論に至るかを検証する手順を定める。

これはデータセットの差し替えではない。37名版の主解析（第13節）は独立に実施し、本節の結果で置き換えない。

### 17.2 2実験は別実験である（設計差の記録）

37名版が先行研究の結果を再現しないことは、検出力や標本の問題ではなく設計差による。テスト期の刺激スケジュールが異なる。

| | 15名版 (NeuroHaze) | 37名版 |
|---|---|---|
| 本試行数 | 400 | 380 |
| 条件 | 単一（CBなし） | 4条件（normal/reverse × CB） |
| 学習期 白cue | 0.20/0.35/0.50/0.65/0.80 | 0.35–0.65 |
| 学習期 赤cue | 0.90 のみ | 0.10 / 0.90 |
| テスト期 白cue | 0.35–0.65 | 0.35–0.65 |
| **テスト期 赤cue** | **0.35–0.65 ＋ 0.90** | **0.35–0.65 のみ** |

15名版ではテスト期を通じて赤cueのみが易しい試行を含み、その比率は3フェーズで安定している（Phase1 0.132 / Phase2 0.113 / Phase3 0.130、白cueは全フェーズ 0.000）。37名版では両cueとも 0.000 であり、テスト期のcue×coherenceは完全均衡である（第5.4節）。

すなわち15名版の赤cueはテスト期中も「約13%の確率で易しい試行」という実効的contingencyを保持していたのに対し、37名版はこのcontingencyを除去している。**先行研究でゲイン変調（M3/M5）が選ばれた機序は15名版の設計に整合し、37名版で帰無となることも同じ理由から予測される。** 両者は矛盾せず、異なる問いに答えている。

なお先行研究は coherence 0.1/0.9 を解析から除外している（`hssm_cue_effect.py:144`）。これにより解析対象試行では両cueの刺激分布が一致し、cueの意味だけが経験によって異なる構造になる。除外された0.90試行は参加者が経験しておりcueに意味を与えている。この区別は第17.6節の `u`/`y` 設計へそのまま対応させる。

### 17.3 段階構成

**段階1（本節の主目的）: 静的cue版によるHSSM等価モデルの実装。** cueを直接のカテゴリ回帰子として扱い、HGFを使わない。モデルを完全に揃えたうえで推定法だけを変え、順位の一致を見る。

**段階2: 信念駆動版（案D + cue別HGF）の追加。** 「連続版」の理論的貢献はここにある。段階1を通過してから着手する。

段階1を先に行う理由は、モデルを揃えずに推定法を変えると、順位の不一致が推定法の差か理論の差か切り分けられないためである。

### 17.4 `ddm_hgf_linear` の仕様

公式 `ddm_hgf.m` および `ddm_hgf_coherence.m` はHSSMのdrift関数形を表現できない。

```text
HSSM:  v = β0 + β_coh · s
PAM  :  v = sign(s) · (a_v + b_c·|s| + b_v·(β−0.5))
```

PAMには `sign(s)·a_v` という s=0 で高さ 2·a_v の不連続段差があり、HSSMにはこの項がない。かつ `a_v = exp(θ) > 0` のためゼロにできない。したがって既存モデルではHSSMのM0すら表現できない。

新規response model `ddm_hgf_linear` を別名で追加する（`ddm_hgf.m` は変更しない。第9.3節の禁止境界を維持）。

**cue符号化**: \(\kappa_r = +1\)（白cue）、\(-1\)（赤cue）。偏差符号化により、prior varianceを0にするだけで下位モデルへ入れ子に縮退する（第2.3節と同じ idiom）。

**coherence尺度**: HSSMと同一の \(\mathtt{coh\_centered} = (ratio_{corrected} - 0.5)\times 10\) を使う。第5.2節の `signed_coherence` は \((ratio-0.5)\times 2\) であり、係数が5倍ずれてパラメータ値を直接比較できないため、本節では使わない。

**試行別パラメータ**:

\[
v_r=(v_0+d_{v0}\kappa_r)+(b_c+d_c\kappa_r)\cdot \mathtt{coh\_centered}_r
\]
\[
w_r=\operatorname{sigmoid}(\theta_w+d_w\kappa_r)\in(0,1)
\]
\[
a_r=a_a=\exp(\theta_a)>0,\qquad Ter=\min(RT_{valid})\operatorname{sigmoid}(\theta_{Ter})
\]

**自由パラメータ**: \(v_0, b_c, d_{v0}, d_c, \theta_w, d_w, \theta_a, \theta_{Ter}\)（すべて無制約空間で推定）。

この形は \(s=0\) でも \(v=v_0+d_{v0}\kappa\) が well-defined であり、第5.2.1節の tie試行（ratio=0.5）問題も同時に解消する。

**知覚モデル**: 段階1ではHGFを使わない。`tapas_ehgf_binary` を置きつつ全知覚パラメータのprior varianceを0にして固定し、自由パラメータを応答モデルのみとする。これによりjoint MAPの対象がHSSMと一致する。ゼロ自由知覚パラメータで `tapas_fitModel` が正常動作するかはスモークで検証し、不可なら恒等knownパラメータのダミー知覚モデルを別名で用意する。

### 17.5 HSSM M0–M7 との対応表

| モデル | HSSM `v` formula | HSSM `z` formula | `ddm_hgf_linear` で自由にする偏差項 |
|---|---|---|---|
| M0_Null | `1 + coh_centered` | `1` | なし |
| M1_Motor | `1 + coh_centered` | `0 + C(cue)` | \(d_w\) |
| M2_PerceptBias | `0 + C(cue) + coh_centered` | `1` | \(d_{v0}\) |
| M3_Gain | `1 + coh_centered + coh_centered:C(cue)` | `1` | \(d_c\) |
| M4_MotorPercept | `0 + C(cue) + coh_centered` | `0 + C(cue)` | \(d_{v0}, d_w\) |
| M5_MotorGain | `1 + coh_centered + coh_centered:C(cue)` | `0 + C(cue)` | \(d_c, d_w\) |
| M6_PerceptGain | `0 + C(cue) + C(cue):coh_centered` | `1` | \(d_{v0}, d_c\) |
| M7_proposed | `0 + C(cue) + C(cue):coh_centered` | `0 + C(cue)` | \(d_{v0}, d_c, d_w\) |

全モデルでHSSMの `a` と `t` はcue非依存（`~ 1 + (1|subject_id)`）である。**HSSMのモデル空間には境界変調が一切含まれない。** 37名版の記述的解析が示唆した反応慎重度の説明（第0節参照）は先行研究では検証されていない。段階2でPAMの `b_a` を使って初めて検証可能になる。

**M4の非識別性**: 先行研究はM4を \(\hat R>1.5\) の収束不良によりLOSO比較から除外している。この非識別性（\(z\) と \(v\) 切片がともに反応バイアスを説明する）は推定法ではなくモデル構造に由来するため、PAMでも同じ病理が出るはずである。PAMでは `optim.Corr` の \(d_{v0}\)–\(d_w\) 相関として直接観測される。**これを段階1の独立した妥当性確認に使う。** 相関が±1近傍にならない場合、実装かデータ整合を疑う。

### 17.6 データ前処理の整合

尤度対象試行を先行研究と1試行単位で一致させ、かつPAMの信念更新に必要な履歴を保持する。第6節の `u`/`y` 設計をそのまま適用する。

| 項目 | HSSM | PAM側の対応 |
|---|---|---|
| coherence 0.1/0.9 | fitから除外 | `u` に残す／`y=[NaN,NaN]` |
| 学習期 1–100 | 未使用 | `u` に残す／`y=[NaN,NaN]` |
| フェーズ分割 | 各フェーズ独立にfit | 対象フェーズ外を `y=[NaN,NaN]`。フェーズごとに別run |
| timeout | 除外 | `y=[NaN,NaN]` |

フェーズ範囲は先行研究に合わせる。Phase1 = 101–200、Phase2 = 201–300、Phase3 = 301–400。

段階1では知覚パラメータが固定でHGFが学習しないため、`u` の保持は段階2への準備であり段階1の結果には影響しない。この不変性自体をテストで確認する（`u` を変えても段階1のMAPが変わらないこと）。

### 17.7 揃えられない差分の記録

以下は設計上残る差であり、結果の解釈時に必ず併記する。

1. **階層構造**: HSSMは被験者ランダム切片を持ち群平均へ縮約する。PAMは被験者ごとに独立なMAPで縮約がない。n=15では縮約の影響は小さくなく、順位が変わりうる既知要因である。
2. **比較指標**: HSSMはPSIS-LOO（試行レベル）とPSIS-LOSO（被験者レベル）。PAMはLaplace LME＋`spm_BMS_gibbs`（被験者レベルrandom-effects）。ΔELPDとexceedance probabilityは直接比較できず、比較できるのは**順位**のみ。
3. **正しいベンチマークはLOOではなくLOSOである**。PAMのrandom-effects BMSはLOSOと同じく被験者集団への一般化を評価する指標であり、試行レベルのLOOとは階層が異なる。先行研究のLOSOは全フェーズ・全モデルで有意差なし（最小 p=0.068）であった。したがって**PAMでも帰無が出ることが手法一致の予測であり、それは段階1の成功を意味する**。
4. **事前分布**: HSSMは `t~Gamma(6,20)`, `a~Gamma(4,3)`, `z~Beta(10,10)`, `v~Normal` を使う。PAMは無制約空間のガウス事前である。等価な事前ではないため、事前の違いに対する感度分析を行う。
5. **先行研究のLOO診断**: 全24行に Pareto-k 警告（`warning=True`）が立っている。LOO点推定の順位自体が不安定である可能性を、比較の前提として記録する。

### 17.8 段階1の合格基準

- 15名×3フェーズ×8モデルが有限のMAP・LME・Hessianを返す。
- M4で \(d_{v0}\)–\(d_w\) の強い相関が観測され、先行研究の \(\hat R\) 由来の除外判断と独立に整合する。
- \(d_{v0}=d_c=d_w=0\) に固定したM0が、`ddm_hgf_linear` の縮退として公式PAMと矛盾しない挙動を示す。
- フェーズ別のrandom-effects BMS結果を、先行研究のLOSO（有意差なし）およびLOO（Motor→Null→Motor+Gain）と並べて報告する。両者が一致しない場合、第17.7節のどの差分に帰属するかを特定してから段階2へ進む。
- 段階1の結果を37名版の主解析（第13節）へ流用しない。

## 18. 参照URL

- PAM repository: <https://github.com/antovis86/PAM-PredictiveAccumulationModels>
- PAM OSF materials: <https://osf.io/3jve9>
- TAPAS archived repository: <https://github.com/translationalneuromodeling/tapas>
- Current HGF organization: <https://github.com/ComputationalPsychiatry>
- SPM12 releases: <https://github.com/spm/spm12>
