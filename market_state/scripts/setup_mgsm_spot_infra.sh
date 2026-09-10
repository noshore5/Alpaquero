#!/bin/bash
# setup_mgsm_spot_infra.sh -- ONE-TIME: create the OIDC launcher role that
# .github/workflows/mgsm-spot-keepalive.yml assumes, so the Alpaquero repo can
# launch spot GPU boxes with NO stored credentials.
#
# Reuses the existing account infra (shared with EEG_Benchmarks):
#   - OIDC provider token.actions.githubusercontent.com   (already exists)
#   - instance profile  eeg-gpu   (its role already has S3 RW on the bucket)
#   - S3 bucket         noshore-eeg-benchmarks-827938107865
#   - SNS topic         eeg-runs
#
# Creates:
#   - IAM role  alpaquero-gh-launcher , assumable ONLY by noshore5/Alpaquero
#     workflows, with an inline policy scoped to: RunInstances + ec2:Describe*
#     + CreateTags-on-launch + Terminate/Stop gated to Project=alpaquero +
#     PassRole eeg-gpu + S3 RW on checkpoints/mgsm-* & datasets/mgsm-* &
#     exports/mgsm/* + sns:Publish on eeg-runs.
#
# Run once, from a shell with admin AWS creds for account 827938107865.
# Idempotent.
set -euo pipefail
REGION=us-east-1
ACCT=827938107865
REPO=noshore5/Alpaquero
ROLE=alpaquero-gh-launcher
BUCKET=noshore-eeg-benchmarks-827938107865
TOPIC="arn:aws:sns:${REGION}:${ACCT}:eeg-runs"
OIDC_HOST=token.actions.githubusercontent.com
OIDC_ARN="arn:aws:iam::${ACCT}:oidc-provider/${OIDC_HOST}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "== GitHub OIDC provider =="
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  echo "  exists: $OIDC_ARN"
else
  aws iam create-open-id-connect-provider --url "https://${OIDC_HOST}" \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
    --query OpenIDConnectProviderArn --output text
fi

echo "== IAM role $ROLE (trusts only repo:$REPO) =="
# this account presents a customised sub: repo:<owner>@<ownerid>/<repo>@<repoid>:...
cat > "$TMP/trust.json" <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Effect": "Allow",
    "Principal": { "Federated": "$OIDC_ARN" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "${OIDC_HOST}:aud": "sts.amazonaws.com",
        "${OIDC_HOST}:repository": "${REPO}"
      },
      "StringLike": { "${OIDC_HOST}:sub": "repo:${REPO%%/*}@*/${REPO#*/}@*:*" }
    } }
] }
JSON
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE" --policy-document "file://$TMP/trust.json"
  echo "  trust policy updated"
else
  aws iam create-role --role-name "$ROLE" \
    --description "GitHub Actions mgsm-spot-keepalive.yml -> launch spot GPU training boxes (OIDC)" \
    --assume-role-policy-document "file://$TMP/trust.json" \
    --query 'Role.Arn' --output text
fi

echo "== inline policy alpaquero-gh-launch =="
cat > "$TMP/pol.json" <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Sid": "Discover", "Effect": "Allow",
    "Action": ["ec2:DescribeImages","ec2:DescribeSecurityGroups",
               "ec2:DescribeInstances","ec2:DescribeInstanceStatus",
               "ec2:DescribeSubnets","ec2:DescribeVpcs"],
    "Resource": "*" },
  { "Sid": "Launch", "Effect": "Allow",
    "Action": "ec2:RunInstances", "Resource": "*" },
  { "Sid": "TagOnLaunch", "Effect": "Allow",
    "Action": "ec2:CreateTags",
    "Resource": "arn:aws:ec2:${REGION}:${ACCT}:*/*",
    "Condition": { "StringEquals": { "ec2:CreateAction": "RunInstances" } } },
  { "Sid": "KillOwnBoxes", "Effect": "Allow",
    "Action": ["ec2:TerminateInstances","ec2:StopInstances"],
    "Resource": "arn:aws:ec2:${REGION}:${ACCT}:instance/*",
    "Condition": { "StringEquals": { "ec2:ResourceTag/Project": "alpaquero" } } },
  { "Sid": "PassInstanceProfileRole", "Effect": "Allow",
    "Action": "iam:PassRole",
    "Resource": "arn:aws:iam::${ACCT}:role/eeg-gpu",
    "Condition": { "StringEquals": { "iam:PassedToService": "ec2.amazonaws.com" } } },
  { "Sid": "ListBucketMgsmPrefixes", "Effect": "Allow",
    "Action": "s3:ListBucket",
    "Resource": "arn:aws:s3:::${BUCKET}",
    "Condition": { "StringLike": { "s3:prefix": ["checkpoints/*","datasets/*","exports/mgsm/*"] } } },
  { "Sid": "RwCheckpointsAndExports", "Effect": "Allow",
    "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject"],
    "Resource": ["arn:aws:s3:::${BUCKET}/checkpoints/*",
                 "arn:aws:s3:::${BUCKET}/exports/mgsm/*"] },
  { "Sid": "ReadDatasets", "Effect": "Allow",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::${BUCKET}/datasets/*" },
  { "Sid": "NotifyOnGiveUp", "Effect": "Allow",
    "Action": "sns:Publish", "Resource": "${TOPIC}" }
] }
JSON
aws iam put-role-policy --role-name "$ROLE" --policy-name alpaquero-gh-launch \
  --policy-document "file://$TMP/pol.json"
echo "  attached"

cat <<DONE

== done ==
Role ARN: arn:aws:iam::${ACCT}:role/${ROLE}
  (already hardcoded in .github/workflows/mgsm-spot-keepalive.yml)

Next:
  1. Build features once and upload:
       cd market_state
       ../.venv_market/bin/python scripts/build_features.py --config configs/crypto.yaml
       aws s3 cp data/features/features.npz \\
         s3://${BUCKET}/datasets/mgsm/features.npz
  2. Merge this branch to main (the */15 cron only runs from the default branch).
  3. Start a run:
       gh workflow run mgsm-spot-keepalive.yml
     ...or with a shorter shakedown command:
       gh workflow run mgsm-spot-keepalive.yml \\
         -f cmd='python market_state/scripts/train.py --config market_state/configs/crypto.yaml --features /root/features.npz --device cuda --max-epochs 20 --patience 5 --max-folds 4'
  4. Watch:
       aws s3 cp s3://${BUCKET}/exports/mgsm/mgsm/run.log -
       aws s3 ls s3://${BUCKET}/checkpoints/mgsm/
  5. Restart a finished/failed run:  add  -f reset=true
DONE
