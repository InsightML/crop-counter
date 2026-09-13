#!/usr/bin/env bash
# EC2 user-data for the CFD-17 benchmark box. cloud-init runs this as ROOT on
# first boot; launch.sh substitutes the four __PLACEHOLDER__ values into a temp
# copy before passing it as --user-data.
#
# Everything after the disk setup runs as the `ubuntu` user, because the DLAMI's
# conda environment and pip caches belong to it.
set -euo pipefail
exec > >(tee -a /var/log/cfd-bootstrap.log) 2>&1

BRANCH="${BRANCH:-__BRANCH__}"
S3_URI="${S3_URI:-__S3_URI__}"
RESUME="${RESUME:-__RESUME__}"
# The code arrives as a `git archive` tarball that launch.sh put in the bucket
# (crop-counter is a private repo, and user-data is readable from inside the
# instance, so no GitHub credential ever travels here).
CODE_KEY="${CODE_KEY:-__CODE_KEY__}"

REGION="${REGION:-__REGION__}"            # where this box runs
S3_REGION="${S3_REGION:-__S3_REGION__}"   # where the bucket lives (us-east-1)
SSM_REGION="${SSM_REGION:-eu-west-2}"     # the MLflow SecureStrings live here
NVME="${NVME:-/opt/dlami/nvme}"

echo "=== cfd bootstrap $(date -u '+%Y-%m-%dT%H:%M:%SZ') branch=$BRANCH code=$CODE_KEY resume=$RESUME ==="

# --- instance store --------------------------------------------------------- #
mkdir -p "$NVME"
if mountpoint -q "$NVME"; then
    echo "$NVME already mounted"
else
    DEV="$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/ {print "/dev/"$1; exit}')"
    if [ -n "${DEV:-}" ]; then
        echo "formatting instance store $DEV -> $NVME"
        mkfs -t ext4 -F "$DEV"
        mount "$DEV" "$NVME"
    else
        echo "no instance-store device found; using the root volume at $NVME"
    fi
fi
chown -R ubuntu:ubuntu "$NVME"
df -h "$NVME"

# The AWS CLI is on every DLAMI, but a box without it must fail here, loudly.
if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI missing — installing v2"
    curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    (cd /tmp && unzip -q awscliv2.zip && ./aws/install) && rm -rf /tmp/aws /tmp/awscliv2.zip
fi
aws --version

# Values are passed through `env`, not sudo's --preserve-env, so no sudoers
# policy is involved. The heredoc is single-quoted: nothing expands host-side.
sudo -u ubuntu -H env \
    BRANCH="$BRANCH" S3_URI="$S3_URI" RESUME="$RESUME" CODE_KEY="$CODE_KEY" \
    REGION="$REGION" S3_REGION="$S3_REGION" SSM_REGION="$SSM_REGION" NVME="$NVME" \
    bash -s <<'UBUNTU'
set -euo pipefail
REPO_DIR="$NVME/crop-counter"

# --- python environment ----------------------------------------------------- #
# The Ubuntu 24.04 DLAMI ships PyTorch in a venv at /opt/pytorch, with CUDA
# INSIDE it (/opt/pytorch/cuda) -- so the venv must be active or there is no GPU.
# The conda branch is the older (22.04) DLAMI layout, kept as a fallback.
if [ -f /opt/pytorch/bin/activate ]; then
    # shellcheck disable=SC1091
    . /opt/pytorch/bin/activate
    DLAMI_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    if [ -w "$DLAMI_SITE" ]; then
        echo "installing into the DLAMI venv ($DLAMI_SITE is writable)"
    else
        # Not writable by ubuntu: make our own venv. NOTE a nested venv's
        # --system-site-packages exposes the BASE interpreter's packages, not
        # the /opt/pytorch venv's, so torch would vanish -- hence the .pth
        # file that puts the DLAMI site-packages (torch + its bundled CUDA)
        # back on sys.path.
        if [ ! -f "$NVME/venv/bin/activate" ]; then
            python -m venv --system-site-packages "$NVME/venv"
        fi
        # shellcheck disable=SC1091
        . "$NVME/venv/bin/activate"
        OUR_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
        echo "$DLAMI_SITE" > "$OUR_SITE/dlami-pytorch.pth"
        echo "installing into $NVME/venv (+ $DLAMI_SITE via .pth)"
    fi
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    . /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch 2>/dev/null || conda activate base
else
    echo "no DLAMI python found -- falling back to a fresh venv"
    python3 -m venv "$NVME/venv"
    # shellcheck disable=SC1091
    . "$NVME/venv/bin/activate"
fi
echo "python: $(command -v python) $(python -V 2>&1)"
# Fatal on purpose: a box without torch+CUDA must die here, in the bootstrap
# log, not 20 minutes later inside run_all.sh.
python -c 'import sys, torch; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda", ok); sys.exit(0 if ok else 1)'

# --- code ------------------------------------------------------------------- #
if [ -f "$REPO_DIR/pyproject.toml" ]; then
    echo "code already unpacked at $REPO_DIR"
else
    mkdir -p "$REPO_DIR"
    aws s3 cp --region "$S3_REGION" "$S3_URI/$CODE_KEY" /tmp/crop-counter-code.tgz
    tar -xzf /tmp/crop-counter-code.tgz -C "$REPO_DIR"
    rm -f /tmp/crop-counter-code.tgz
fi
echo "code: $CODE_KEY (branch $BRANCH)"
( cd "$REPO_DIR" && pip install -q -e ".[dev,portal,detection]" rfdetr supervision mlflow boto3 )

# --- inputs ----------------------------------------------------------------- #
mkdir -p "$REPO_DIR/weights" "$REPO_DIR/data/cfd" "$NVME/runs" "$NVME/results" "$NVME/cfd17"
for f in dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth decoder_best.pt; do
    [ -f "$REPO_DIR/weights/$f" ] || \
        aws s3 cp --region "$S3_REGION" "$S3_URI/inputs/$f" "$REPO_DIR/weights/$f"
done
META=community_fish_detection_dataset.json.zip
[ -f "$REPO_DIR/data/cfd/$META" ] || \
    aws s3 cp --region "$S3_REGION" "$S3_URI/inputs/$META" "$REPO_DIR/data/cfd/$META"

# --- resume ----------------------------------------------------------------- #
if [ "$RESUME" = "1" ]; then
    echo "resuming: pulling runs/, results/ and the subset metadata back down"
    # Fatal if these fail: a resume that silently starts fresh would then
    # overwrite the checkpoints on S3 with an inferior run.
    aws s3 sync --region "$S3_REGION" "$S3_URI/runs" "$NVME/runs" --only-show-errors
    aws s3 sync --region "$S3_REGION" "$S3_URI/results" "$NVME/results" --only-show-errors
    # The COCO jsons + subset_summary, so the streaming subset pass is skipped.
    # The pixels are NOT restored: the instance store is empty, so fetch re-pulls
    # them, and the .fetched marker is deliberately removed afterwards.
    aws s3 sync --region "$S3_REGION" "$S3_URI/data-manifest" "$NVME/cfd17" --only-show-errors || true
    rm -f "$NVME/cfd17/.fetched"
    # A run dir that came back without a .done marker resumes from its last.pt.
fi

# --- MLflow credentials (never echoed) -------------------------------------- #
MLFLOW_PW="$(aws ssm get-parameter --region "$SSM_REGION" --with-decryption \
    --name /insightml/prod/mlflow/user-freddie-password \
    --query Parameter.Value --output text 2>/dev/null || true)"
if [ -n "$MLFLOW_PW" ] && [ "$MLFLOW_PW" != "None" ]; then
    export MLFLOW_TRACKING_URI=https://mlflow.insightml.io
    export MLFLOW_TRACKING_USERNAME=freddie
    export MLFLOW_TRACKING_PASSWORD="$MLFLOW_PW"
    echo "MLflow credentials loaded from SSM ($SSM_REGION)"
else
    echo "no MLflow password from SSM -- logging will be skipped (not fatal)"
fi
unset MLFLOW_PW

# --- go --------------------------------------------------------------------- #
export DATA_ROOT="$NVME/cfd17" RUNS_DIR="$NVME/runs" RESULTS_DIR="$NVME/results"
export S3_URI DEVICE=cuda
# run_all.sh's `aws s3 sync` calls carry no --region: the high-level s3 commands
# resolve the bucket's region themselves, but a default region must exist.
export AWS_DEFAULT_REGION="$S3_REGION"
# setsid: a new session, so the end of cloud-init's unit cannot take the run
# with it; stdin from /dev/null so the child never reads this heredoc.
setsid nohup bash "$REPO_DIR/examples/FishDetection/scripts/run_all.sh" \
    < /dev/null > "$NVME/run_all.nohup" 2>&1 &
echo "run_all.sh started (pid $!); log at $NVME/runs/run_all.log"
UBUNTU

echo "=== cfd bootstrap handed off $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
