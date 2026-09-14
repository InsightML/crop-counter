#!/usr/bin/env bash
# What is the benchmark box doing? Instance state + the tail of run_all.log +
# GPU utilisation, fetched over SSM (the SG has no ingress, so there is no ssh).
#
#   ./status.sh          last 80 log lines
#   ./status.sh 1000     last 1000
set -euo pipefail

REGION="${REGION:-us-east-1}"
S3_REGION="${S3_REGION:-us-east-1}"
BUCKET="${BUCKET:-insightml-cfd-benchmark}"
S3_URI="${S3_URI:-s3://$BUCKET}"
NVME="${NVME:-/opt/dlami/nvme}"
LINES="${1:-80}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
ID_FILE="$REPO/.aws-instance-id"

[ -s "$ID_FILE" ] || { echo "no instance id at $ID_FILE — has launch.sh run?" >&2; exit 1; }
INSTANCE_ID="$(tr -d '[:space:]' < "$ID_FILE")"

aws_ec2() { aws ec2 --region "$REGION" "$@"; }
aws_ssm() { aws ssm --region "$REGION" "$@"; }

read -r STATE LAUNCHED TYPE <<<"$(aws_ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].[State.Name,LaunchTime,InstanceType]' \
    --output text)"
echo "instance   $INSTANCE_ID  $TYPE  state=$STATE"
echo "launched   $LAUNCHED"
if command -v python3 >/dev/null 2>&1; then
    python3 - "$LAUNCHED" <<'PY'
import datetime, sys
started = datetime.datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
delta = datetime.datetime.now(datetime.timezone.utc) - started
hours, rem = divmod(int(delta.total_seconds()), 3600)
print(f"uptime     {hours}h {rem // 60}m")
PY
fi

if [ "$STATE" != "running" ]; then
    echo "instance is not running — skipping the SSM probe"
else
    # The remote command is base64'd so quoting cannot be mangled on the way in.
    # GPU and disk first: GetCommandInvocation truncates stdout at 24,000
    # characters, so a long log tail must be the thing that gets cut.
    REMOTE="$(cat <<REMOTE_EOF
echo '--- gpu ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo '(no nvidia-smi)'
echo '--- disk ---'
df -h $NVME 2>/dev/null | tail -n 1
echo '--- cloud-init bootstrap tail ---'
tail -c 20000 /var/log/cfd-bootstrap.log 2>/dev/null | tr '\\r' '\\n' | grep -v '^Completed' | tail -n 15 | cut -c1-300 || echo '(no bootstrap log yet)'
echo '--- run_all.log (last $LINES) ---'
tail -n $LINES $NVME/runs/run_all.log 2>/dev/null | cut -c1-400 || echo '(no log yet)'
REMOTE_EOF
)"
    B64="$(printf '%s' "$REMOTE" | base64 | tr -d '\n')"
    CMD_ID="$(aws_ssm send-command --instance-ids "$INSTANCE_ID" \
        --document-name AWS-RunShellScript \
        --parameters "commands=echo $B64 | base64 -d | bash" \
        --query 'Command.CommandId' --output text)"
    echo "ssm command $CMD_ID — waiting..."
    for _ in $(seq 1 60); do
        SSM_STATE="$(aws_ssm get-command-invocation --command-id "$CMD_ID" \
            --instance-id "$INSTANCE_ID" --query 'Status' --output text 2>/dev/null || echo Pending)"
        case "$SSM_STATE" in
            Pending|InProgress|Delayed) sleep 2 ;;
            *) break ;;
        esac
    done
    echo "ssm status  $SSM_STATE"
    aws_ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
        --query 'StandardOutputContent' --output text || true
    aws_ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
        --query 'StandardErrorContent' --output text >&2 || true
fi

echo "--- synced history.json in $S3_URI/runs ---"
aws s3 ls --region "$S3_REGION" --recursive "$S3_URI/runs" | grep history.json || echo "(none yet)"
