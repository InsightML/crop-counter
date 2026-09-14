#!/usr/bin/env bash
# One-time AWS scaffolding for the CFD-17 benchmark: bucket, IAM role + instance
# profile, an SSM-only security group, and the run's input files staged in S3.
#
# Idempotent throughout — every create is preceded by a describe/get, so running
# this twice changes nothing. Safe to re-run after a partial failure.
set -euo pipefail

REGION="${REGION:-us-east-1}"
# The bucket stays in us-east-1 even when the box runs elsewhere (S3 is global-
# named; the data is small enough that cross-region reads do not matter).
S3_REGION="${S3_REGION:-us-east-1}"
# The MLflow SecureStrings live in eu-west-2; everything else is us-east-1.
SSM_REGION="${SSM_REGION:-eu-west-2}"
ACCOUNT="${ACCOUNT:-944269089535}"
BUCKET="${BUCKET:-insightml-cfd-benchmark}"
ROLE="${ROLE:-cfd-benchmark-ec2}"
PROFILE_NAME="${PROFILE_NAME:-cfd-benchmark-ec2}"
SG_NAME="${SG_NAME:-cfd-benchmark-ssm-only}"
PROJECT_TAG="${PROJECT_TAG:-cfd-benchmark}"
LIFECYCLE_DAYS="${LIFECYCLE_DAYS:-90}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"

aws_ec2() { aws ec2 --region "$REGION" "$@"; }
aws_s3api() { aws s3api --region "$S3_REGION" "$@"; }

echo "== S3 bucket s3://$BUCKET =="
if aws_s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    echo "   exists"
else
    # us-east-1 is the one region where create-bucket must NOT get a
    # LocationConstraint; this whole stack is us-east-1, so no branch is needed.
    if [ "$S3_REGION" = "us-east-1" ]; then
        aws_s3api create-bucket --bucket "$BUCKET"
    else
        aws_s3api create-bucket --bucket "$BUCKET" \
            --create-bucket-configuration "LocationConstraint=$S3_REGION"
    fi
    echo "   created"
fi

aws_s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration \
    'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'
aws_s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration 'Status=Suspended'
aws_s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" \
    --lifecycle-configuration "{\"Rules\":[{\"ID\":\"expire-${LIFECYCLE_DAYS}d\",\"Status\":\"Enabled\",\"Filter\":{\"Prefix\":\"\"},\"Expiration\":{\"Days\":${LIFECYCLE_DAYS}},\"AbortIncompleteMultipartUpload\":{\"DaysAfterInitiation\":7}}]}"
aws_s3api put-bucket-tagging --bucket "$BUCKET" \
    --tagging "TagSet=[{Key=project,Value=$PROJECT_TAG}]"
echo "   public access blocked | versioning off | objects expire after ${LIFECYCLE_DAYS}d"

echo "== IAM role $ROLE =="
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    echo "   exists"
else
    aws iam create-role --role-name "$ROLE" \
        --assume-role-policy-document "$TRUST" \
        --tags "Key=project,Value=$PROJECT_TAG" >/dev/null
    echo "   created"
fi

# S3 on this bucket only, plus the one MLflow password parameter (SecureString
# under the AWS-managed alias/aws/ssm key, which needs no extra kms grant).
INLINE="{\"Version\":\"2012-10-17\",\"Statement\":[\
{\"Sid\":\"BucketList\",\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\",\"s3:GetBucketLocation\"],\"Resource\":\"arn:aws:s3:::${BUCKET}\"},\
{\"Sid\":\"Objects\",\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:AbortMultipartUpload\",\"s3:ListMultipartUploadParts\"],\"Resource\":\"arn:aws:s3:::${BUCKET}/*\"},\
{\"Sid\":\"MlflowPassword\",\"Effect\":\"Allow\",\"Action\":[\"ssm:GetParameter\"],\"Resource\":\"arn:aws:ssm:${SSM_REGION}:${ACCOUNT}:parameter/insightml/prod/mlflow/user-freddie-password\"}]}"
aws iam put-role-policy --role-name "$ROLE" \
    --policy-name "${ROLE}-inline" --policy-document "$INLINE"
aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
echo "   inline policy + AmazonSSMManagedInstanceCore attached"

echo "== instance profile $PROFILE_NAME =="
if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
    echo "   exists"
else
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" \
        --tags "Key=project,Value=$PROJECT_TAG" >/dev/null
    echo "   created (waiting 10s for IAM propagation so a launch right after this works)"
    sleep 10
fi
if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" \
        --query "InstanceProfile.Roles[?RoleName=='$ROLE'].RoleName" --output text \
        | grep -q "$ROLE"; then
    echo "   role already attached"
else
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE"
    echo "   role attached"
fi
PROFILE_ARN="$(aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" \
    --query 'InstanceProfile.Arn' --output text)"

echo "== security group $SG_NAME =="
VPC_ID="${VPC_ID:-$(aws_ec2 describe-vpcs --filters 'Name=isDefault,Values=true' \
    --query 'Vpcs[0].VpcId' --output text)}"
SG_ID="$(aws_ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)"
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID="$(aws_ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
        --description 'CFD benchmark: no ingress, SSM egress only' \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$SG_NAME},{Key=project,Value=$PROJECT_TAG}]" \
        --query 'GroupId' --output text)"
    echo "   created $SG_ID in $VPC_ID"
    # A new SG ships with allow-all egress; replace it with the narrow set.
    aws_ec2 revoke-security-group-egress --group-id "$SG_ID" \
        --ip-permissions 'IpProtocol=-1,IpRanges=[{CidrIp=0.0.0.0/0}]' >/dev/null 2>&1 || true
else
    echo "   exists $SG_ID"
fi

# No ingress at all: the box is reached through SSM Session Manager, which is an
# outbound connection from the agent. Egress: 443 (SSM/S3/PyPI/GitHub/MLflow),
# 53 (DNS), 123/udp (NTP), 80 (apt).
add_egress() {
    # Idempotent: only a duplicate-rule error is ignored; anything else is fatal
    # (a swallowed failure would leave the box with no egress and SSM unreachable).
    local err
    if err="$(aws_ec2 authorize-security-group-egress --group-id "$SG_ID" \
            --ip-permissions "$1" 2>&1 >/dev/null)"; then
        return 0
    fi
    grep -q 'InvalidPermission.Duplicate' <<<"$err" || { echo "$err" >&2; return 1; }
}
add_egress 'IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description=https}]'
add_egress 'IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description=apt}]'
add_egress 'IpProtocol=tcp,FromPort=53,ToPort=53,IpRanges=[{CidrIp=0.0.0.0/0,Description=dns-tcp}]'
add_egress 'IpProtocol=udp,FromPort=53,ToPort=53,IpRanges=[{CidrIp=0.0.0.0/0,Description=dns-udp}]'
add_egress 'IpProtocol=udp,FromPort=123,ToPort=123,IpRanges=[{CidrIp=0.0.0.0/0,Description=ntp}]'
echo "   egress 443/80/53(tcp+udp)/123(udp); no ingress"

echo "== staging inputs to s3://$BUCKET/inputs/ =="
stage() {
    local local_path="$1" key="inputs/$(basename "$1")"
    if [ ! -f "$local_path" ]; then
        echo "   MISSING $local_path — the box cannot run without it" >&2
        return 1
    fi
    local size remote
    size="$(wc -c < "$local_path" | tr -d ' ')"
    remote="$(aws_s3api head-object --bucket "$BUCKET" --key "$key" \
        --query 'ContentLength' --output text 2>/dev/null || echo none)"
    if [ "$remote" = "$size" ]; then
        echo "   up to date: $key ($size bytes)"
        return 0
    fi
    echo "   uploading $key ($size bytes)"
    aws s3 cp --region "$S3_REGION" "$local_path" "s3://$BUCKET/$key" --only-show-errors
}
stage "$REPO/weights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
stage "$REPO/weights/decoder_best.pt"
stage "$REPO/data/cfd/community_fish_detection_dataset.json.zip"

echo
echo "instance profile ARN : $PROFILE_ARN"
echo "security group id    : $SG_ID  (vpc $VPC_ID)"
echo "bucket               : s3://$BUCKET"
