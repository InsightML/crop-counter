#!/usr/bin/env bash
# Launch the CFD-17 benchmark box: one p5.4xlarge spot instance running
# bootstrap.sh as user-data. Writes the instance id to .aws-instance-id at the
# repo root (gitignored) for status.sh / teardown.sh to pick up.
#
#   ./launch.sh              fresh run
#   ./launch.sh --resume     pull runs/ + results/ back from S3 and carry on
#   ./launch.sh --dry-run    permissions check only, nothing is created
#   ./launch.sh --force      launch even though .aws-instance-id names a live box
#   ./launch.sh --on-demand  no spot market options (full rate; use when spot is exhausted)
set -euo pipefail

REGION="${REGION:-us-east-1}"
S3_REGION="${S3_REGION:-us-east-1}"     # the bucket's region (see setup.sh)
BUCKET="${BUCKET:-insightml-cfd-benchmark}"
S3_URI="${S3_URI:-s3://$BUCKET}"
PROFILE_NAME="${PROFILE_NAME:-cfd-benchmark-ec2}"
SG_NAME="${SG_NAME:-cfd-benchmark-ssm-only}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p5.4xlarge}"
# Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04) 20260827.
# For the current id of this family:
#   aws ssm get-parameter --region us-east-1 \
#     --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.12-ubuntu-24.04/latest/ami-id \
#     --query Parameter.Value --output text
AMI_ID="${AMI_ID:-ami-0a4870b172edcb0f2}"
ROOT_GB="${ROOT_GB:-100}"
BRANCH="${BRANCH:-poc/detection-head-unfrozen}"
PROJECT_TAG="${PROJECT_TAG:-cfd-benchmark}"
NAME_TAG="${NAME_TAG:-cfd-benchmark}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
SUBNET_ID="${SUBNET_ID:-}"
# Forwarded to run_all.sh on the box. Empty = run_all.sh's own defaults.
#   RUNS="name=config [name=config ...]"  which training runs, in order
#   RUN_BASELINES=0                       skip the published RF-DETR baselines
#   MEASURE_SIZES="tiny small"            trunk measurements after training
#   EPOCHS=8                              overrides every run's epoch count
RUNS="${RUNS:-}"
RUN_BASELINES="${RUN_BASELINES:-}"
MEASURE_SIZES="${MEASURE_SIZES:-}"
EPOCHS="${EPOCHS:-}"
# Hours after boot at which the box terminates itself, whatever it is doing.
# Nothing else ever shuts it down. 0 disables (and says so in the boot log).
MAX_WALL_HOURS="${MAX_WALL_HOURS:-10}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
ID_FILE="$REPO/.aws-instance-id"

RESUME=0
DRY_RUN=""
FORCE=0
ON_DEMAND=0
while [ $# -gt 0 ]; do
    case "$1" in
        --resume) RESUME=1 ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        --force) FORCE=1 ;;
        --on-demand) ON_DEMAND=1 ;;   # spot capacity exhausted; costs the full rate
        -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

aws_ec2() { aws ec2 --region "$REGION" "$@"; }

if [ -s "$ID_FILE" ] && [ -z "$DRY_RUN" ]; then
    OLD_ID="$(tr -d '[:space:]' < "$ID_FILE")"
    OLD_STATE="$(aws_ec2 describe-instances --instance-ids "$OLD_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo gone)"
    if [ "$OLD_STATE" != "terminated" ] && [ "$OLD_STATE" != "gone" ] && [ "$FORCE" != "1" ]; then
        echo "refusing to launch: $ID_FILE holds $OLD_ID which is still '$OLD_STATE'." >&2
        echo "run teardown.sh first, or pass --force to launch a second box anyway." >&2
        exit 1
    fi
fi

PROFILE_ARN="$(aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" \
    --query 'InstanceProfile.Arn' --output text)"
VPC_ID="$(aws_ec2 describe-vpcs --filters 'Name=isDefault,Values=true' \
    --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(aws_ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text)"
ROOT_DEVICE="$(aws_ec2 describe-images --image-ids "$AMI_ID" \
    --query 'Images[0].RootDeviceName' --output text)"
if [ -z "$PROFILE_ARN" ] || [ "$PROFILE_ARN" = "None" ] \
   || [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "setup.sh has not run: profile=$PROFILE_ARN sg=$SG_ID" >&2
    exit 1
fi
echo "ami          $AMI_ID (root $ROOT_DEVICE, ${ROOT_GB}GB gp3)"
echo "profile      $PROFILE_ARN"
echo "sg           $SG_ID ($VPC_ID)"
echo "branch       $BRANCH @ $(git -C "$REPO" rev-parse --short HEAD) | resume=$RESUME | s3=$S3_URI"
echo "self-destruct ${MAX_WALL_HOURS}h after boot (MAX_WALL_HOURS=0 disables)"

# The code goes up as a `git archive` of HEAD (crop-counter is private; no
# GitHub credential ever reaches the box). HEAD only -- so refuse a dirty tree,
# or the box would silently run something other than what is committed.
SHA="$(git -C "$REPO" rev-parse --short HEAD)"
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] && [ "$ALLOW_DIRTY" != "1" ]; then
    echo "working tree has uncommitted changes; commit them (the box runs HEAD=$SHA) or set ALLOW_DIRTY=1" >&2
    exit 1
fi
CODE_KEY="code/crop-counter-${SHA}.tgz"
if [ -z "$DRY_RUN" ]; then
    if aws s3api head-object --region "$S3_REGION" --bucket "$BUCKET" --key "$CODE_KEY" >/dev/null 2>&1; then
        echo "code         $CODE_KEY already in the bucket"
    else
        CODE_TGZ="$(mktemp -t cfd-code.XXXXXX)"
        git -C "$REPO" archive --format=tar.gz -o "$CODE_TGZ" HEAD
        aws s3 cp --region "$S3_REGION" "$CODE_TGZ" "s3://$BUCKET/$CODE_KEY" --only-show-errors
        rm -f "$CODE_TGZ"
        echo "code         uploaded $CODE_KEY"
    fi
fi

# user-data is a temp copy of bootstrap.sh with the placeholders filled in.
USER_DATA="$(mktemp -t cfd-bootstrap.XXXXXX)"
trap 'rm -f "$USER_DATA"' EXIT
# A value going into a sed REPLACEMENT must have \, & and the | delimiter
# escaped — RUNS carries filesystem paths, so this is not hypothetical.
sed_escape() { printf '%s' "$1" | sed -e 's/[\\&|]/\\\\&/g'; }
sed -e "s|__BRANCH__|$(sed_escape "$BRANCH")|g" \
    -e "s|__S3_URI__|$(sed_escape "$S3_URI")|g" \
    -e "s|__RESUME__|$(sed_escape "$RESUME")|g" \
    -e "s|__CODE_KEY__|$(sed_escape "$CODE_KEY")|g" \
    -e "s|__REGION__|$(sed_escape "$REGION")|g" \
    -e "s|__S3_REGION__|$(sed_escape "$S3_REGION")|g" \
    -e "s|__RUNS__|$(sed_escape "$RUNS")|g" \
    -e "s|__RUN_BASELINES__|$(sed_escape "$RUN_BASELINES")|g" \
    -e "s|__MEASURE_SIZES__|$(sed_escape "$MEASURE_SIZES")|g" \
    -e "s|__EPOCHS__|$(sed_escape "$EPOCHS")|g" \
    -e "s|__MAX_WALL_HOURS__|$(sed_escape "$MAX_WALL_HOURS")|g" \
    "$HERE/bootstrap.sh" > "$USER_DATA"
# A placeholder left behind means a value this launcher does not know about:
# the box would read the literal "__NAME__" and do the wrong thing silently.
if grep -q '__[A-Z0-9_]\+__' "$USER_DATA"; then
    echo "refusing to launch: unsubstituted placeholder(s) in user-data:" >&2
    grep -o '__[A-Z0-9_]\+__' "$USER_DATA" | sort -u >&2
    exit 1
fi

RUN_ARGS=(
    run-instances
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --count 1
    --iam-instance-profile "Arn=$PROFILE_ARN"
    --security-group-ids "$SG_ID"
    --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEVICE\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
    --user-data "file://$USER_DATA"
    --instance-initiated-shutdown-behavior terminate
    --metadata-options 'HttpTokens=required,HttpEndpoint=enabled'
    --tag-specifications
    "ResourceType=instance,Tags=[{Key=Name,Value=$NAME_TAG},{Key=project,Value=$PROJECT_TAG}]"
    "ResourceType=volume,Tags=[{Key=Name,Value=$NAME_TAG},{Key=project,Value=$PROJECT_TAG}]"
)
if [ "$ON_DEMAND" = "1" ]; then
    echo "on-demand: no spot market options — full rate applies" >&2
else
    RUN_ARGS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi
[ -n "$SUBNET_ID" ] && RUN_ARGS+=(--subnet-id "$SUBNET_ID")
[ -n "$DRY_RUN" ] && RUN_ARGS+=("$DRY_RUN")

if [ -n "$DRY_RUN" ]; then
    # A successful dry run is reported as the DryRunOperation *error* (non-zero
    # exit), so capture the output first -- under pipefail a pipe would carry
    # that exit status past a matching grep.
    DRY_OUT="$(aws_ec2 "${RUN_ARGS[@]}" 2>&1 || true)"
    echo "$DRY_OUT" >&2
    if grep -q 'DryRunOperation' <<<"$DRY_OUT"; then
        echo "dry run OK — the caller has permission to launch this instance"
        exit 0
    fi
    echo "dry run FAILED — see the error above" >&2
    exit 1
fi

INSTANCE_ID="$(aws_ec2 "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)"
printf '%s\n' "$INSTANCE_ID" > "$ID_FILE"
echo "launched $INSTANCE_ID -> $ID_FILE"
echo "watch it with: $HERE/status.sh 200"
