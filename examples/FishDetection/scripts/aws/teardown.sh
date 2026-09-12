#!/usr/bin/env bash
# Stop paying for the benchmark box. Terminates the instance in .aws-instance-id.
#
# The BUCKET IS ALWAYS KEPT — it holds the run outputs, and its own 90-day
# lifecycle rule expires them.
#
#   ./teardown.sh          terminate the instance (asks first)
#   ./teardown.sh --yes    no prompt
#   ./teardown.sh --all    also delete the SG, role and instance profile
set -euo pipefail

REGION="${REGION:-us-east-1}"
BUCKET="${BUCKET:-insightml-cfd-benchmark}"
ROLE="${ROLE:-cfd-benchmark-ec2}"
PROFILE_NAME="${PROFILE_NAME:-cfd-benchmark-ec2}"
SG_NAME="${SG_NAME:-cfd-benchmark-ssm-only}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
ID_FILE="$REPO/.aws-instance-id"

ASSUME_YES=0
DELETE_ALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y) ASSUME_YES=1 ;;
        --all) DELETE_ALL=1 ;;
        -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

aws_ec2() { aws ec2 --region "$REGION" "$@"; }

confirm() {
    [ "$ASSUME_YES" -eq 1 ] && return 0
    printf '%s [y/N] ' "$1"
    read -r reply
    case "$reply" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

if [ -s "$ID_FILE" ]; then
    INSTANCE_ID="$(tr -d '[:space:]' < "$ID_FILE")"
    STATE="$(aws_ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo gone)"
    echo "instance $INSTANCE_ID is $STATE"
    if [ "$STATE" = "gone" ] || [ "$STATE" = "terminated" ]; then
        echo "nothing to terminate"
        rm -f "$ID_FILE"
    elif confirm "terminate $INSTANCE_ID?"; then
        aws_ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
            --query 'TerminatingInstances[0].[InstanceId,CurrentState.Name]' --output text
        echo "waiting for terminated..."
        aws_ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
        rm -f "$ID_FILE"
        echo "terminated; removed $ID_FILE"
    else
        echo "left running"
    fi
else
    echo "no instance id at $ID_FILE — nothing to terminate"
fi

if [ "$DELETE_ALL" -eq 1 ]; then
    if confirm "also delete SG $SG_NAME, role $ROLE and profile $PROFILE_NAME? (bucket kept)"; then
        VPC_ID="$(aws_ec2 describe-vpcs --filters 'Name=isDefault,Values=true' \
            --query 'Vpcs[0].VpcId' --output text)"
        SG_ID="$(aws_ec2 describe-security-groups \
            --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
            --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)"
        if [ -n "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
            aws_ec2 delete-security-group --group-id "$SG_ID" && echo "deleted sg $SG_ID"
        fi
        # A role cannot be deleted while it is in an instance profile, and a
        # profile cannot be deleted while it holds a role.
        aws iam remove-role-from-instance-profile \
            --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE" 2>/dev/null || true
        aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null \
            && echo "deleted instance profile $PROFILE_NAME" || true
        aws iam detach-role-policy --role-name "$ROLE" \
            --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
        aws iam delete-role-policy --role-name "$ROLE" \
            --policy-name "${ROLE}-inline" 2>/dev/null || true
        aws iam delete-role --role-name "$ROLE" 2>/dev/null \
            && echo "deleted role $ROLE" || true
    fi
fi

echo "bucket s3://$BUCKET kept (90-day lifecycle handles the objects)"
