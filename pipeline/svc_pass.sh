#!/bin/bash
# Outer calibration pass on the service instance.
# usage: svc_pass.sh <prev_dir> <new_dir> <alpha_S>
set -e
P=$1; D=$2; a=$3; V=/home/claude/rebuild/v2026; C=/home/claude/rebuild/code2026; T=/home/claude/rebuild/tools
rm -rf $D; cp -r $P $D; rm -f $D/truth.npz $D/clear.npz $D/DONE $D/*.log
cd $D
SIGMA_FILE=$P/supply_at_tau0.csv THETA_INIT=$P/theta_calibrated.npy python $V/fit_station_ascs_sigma.py > fit.log 2>&1
python $V/apply_calibration.py > apply.log 2>&1
export ALPHA_S=$a
python $T/gen_truth.py $D $C $D/truth.npz 1e-6 400 > gen.log 2>&1
python $T/make_gt_files.py $D $C $D/truth.npz > gt.log 2>&1
echo done > DONE
