#!/bin/bash
# =============================================================================
# Submit the whole CARFAC sweep: one Slurm job per run, all in parallel.
#
#   bash Trainings/carfac_sweep/submit_sweep.sh --check     # Validate every run's design; submit nothing
#   bash Trainings/carfac_sweep/submit_sweep.sh             # Validate, then submit every run
#   bash Trainings/carfac_sweep/submit_sweep.sh vel_005 agc_off   # Only these runs (by label)
#
# Run from the repo root, on a login node. Each run gets its own CARFAC config (and cache),
# so no two jobs write the same cache folder. Logs: logs/sw_<label>_<jobid>.out
# Collect the results: grep -h "^SWEEP" logs/sw_*.out
# =============================================================================
set -euo pipefail

# label | sweep.py arguments (empty = the reference configuration)
RUNS=(
    "reference      |"
    "logmel         | --frontend logmel"
    "agc_off        | --open-loop"
    "agc_tc_x2      | --agc-tc-scale 2"
    "agc_tc_x05     | --agc-tc-scale 0.5 --agc-decimation 3 2 2 2"
    "agc_dec3_ctrl  | --agc-decimation 3 2 2 2"
    "vel_005        | --velocity-scale 0.05"
    "vel_020        | --velocity-scale 0.2"
    "minzeta_006    | --min-zeta 0.06"
    "minzeta_015    | --min-zeta 0.15"
    "maxzeta_025    | --max-zeta 0.25"
    "maxzeta_050    | --max-zeta 0.5"
    "erb_050        | --erb-per-step 0.5"
    "erb_033        | --erb-per-step 0.33"
    "taulpf_4e-5    | --tau-lpf 4e-5"
    "taulpf_1.6e-4  | --tau-lpf 1.6e-4"
    "cuberoot       | --post cuberoot"
)

PY="${PY:-/home/ozi0609/.conda/envs/asr1/bin/python}"       # Override: PY=/path/to/python bash ...
SCRIPT=Trainings/carfac_sweep/sweep.py
CHECK_ONLY=0
WANTED=()
for arg in "$@"; do
    if [[ "$arg" == "--check" ]]; then CHECK_ONLY=1; else WANTED+=("$arg"); fi
done

mkdir -p logs

# 1. Validate every selected run (seconds each, CPU only): a bad design fails here, not in the queue
selected=()
for entry in "${RUNS[@]}"; do
    label=$(echo "${entry%%|*}" | xargs)              # Text before "|", trimmed
    args=$(echo "${entry#*|}" | xargs)                # Text after "|", trimmed
    if (( ${#WANTED[@]} )) && [[ ! " ${WANTED[*]} " =~ " ${label} " ]]; then
        continue                                      # Not one of the runs asked for
    fi
    echo "== check $label: ${args:-(reference)}"
    # shellcheck disable=SC2086 (word splitting of $args is intended)
    if ! "$PY" "$SCRIPT" --dry-run $args 2>&1 | grep -E "^RUN|config|CARFAC OK"; then
        echo "   FAILED: fix this run before submitting"; exit 1
    fi
    selected+=("$label|$args")
done

if (( CHECK_ONLY )); then
    echo "All ${#selected[@]} run(s) valid. Submit with: bash $0 ${WANTED[*]}"
    exit 0
fi

# 2. Submit one job per run, named after its label
for entry in "${selected[@]}"; do
    label="${entry%%|*}"; args="${entry#*|}"
    # shellcheck disable=SC2086
    jobid=$(sbatch --parsable --job-name="sw_${label}" Trainings/carfac_sweep/sweep.sbatch $args)
    echo "submitted sw_${label}: job $jobid"
done
echo "Watch: squeue -u \$USER    Logs: tail -f logs/sw_<label>_<jobid>.out"