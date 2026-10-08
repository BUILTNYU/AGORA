#!/bin/bash
# Two supply scales. usage: svc2_run.sh <alpha_LIRR> <alpha_LOCAL_PARK> <tag> [src_dir]
aR=$1; aL=$2; tag=$3; SRC=${4:-/home/claude/rebuild/svc/base}
D=/home/claude/rebuild/svc/$tag; C=/home/claude/rebuild/code2026; T=/home/claude/rebuild/tools
rm -rf $D; cp -r $SRC $D; rm -f $D/truth.npz $D/clear.npz $D/DONE $D/services_at_tau0.csv $D/*.log
python -c "import pandas as pd; pd.DataFrame(dict(operator=['LIRR','LOCAL','PARK'],F=[179520.0,88948.18,0.995*62640],c_idle=[0.27,0.72,0.27],alpha_S=[$aR,$aL,$aL],aS_group=[0,1,1])).to_csv('$D/operator_budgets.csv',index=False)"
python $T/gen_truth.py $D $C $D/truth.npz 1e-6 400 > $D/gen.log 2>&1
python $T/make_gt_files.py $D $C $D/truth.npz > $D/gt.log 2>&1
[ -z "$NOCLEAR" ] && python $T/clear_truth.py $D $C $D/clear.npz > $D/clear.log 2>&1
echo done > $D/DONE
