# Create a clean deployment script without emojis
@'
# factr.ai FREE TIER AWS Deployment Script
Write-Host "FREE TIER factr.ai AWS Deployment" -ForegroundColor Green
Write-Host "Cost: $0.00/month for first 12 months!" -ForegroundColor Cyan

# Step 1: Verify AWS Setup
Write-Host ""
Write-Host "Step 1: Verifying AWS Setup..." -ForegroundColor Yellow

try {
    $awsIdentity = aws sts get-caller-identity | ConvertFrom-Json
    Write-Host "AWS CLI working!" -ForegroundColor Green
    Write-Host "Account: $($awsIdentity.Account)"
} catch {
    Write-Host "AWS CLI not configured!" -ForegroundColor Red
    Write-Host "Run: aws configure"
    exit 1
}

# Set region
$region = aws configure get region
if (-not $region) {
    aws configure set region us-east-1
    $region = "us-east-1"
}
Write-Host "Using region: $region" -ForegroundColor Green

# Step 2: Create S3 Bucket
Write-Host ""
Write-Host "Step 2: Creating S3 Bucket..." -ForegroundColor Yellow

$BUCKET_NAME = "factr-ai-free-$($awsIdentity.Account)-$(Get-Date -Format 'yyyyMMdd')"
Write-Host "Creating bucket: $BUCKET_NAME"

try {
    aws s3 mb s3://$BUCKET_NAME
    Write-Host "S3 bucket created!" -ForegroundColor Green
} catch {
    Write-Host "Failed to create S3 bucket!" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)"
    exit 1
}

# Check if CloudFormation template exists
if (-not (Test-Path "aws_deployment_free.yaml")) {
    Write-Host "aws_deployment_free.yaml not found!" -ForegroundColor Red
    Write-Host "Creating a basic template..."
    
    # Create a minimal CloudFormation template
    $template = @'
AWSTemplateFormatVersion: '2010-09-09'
Description: 'factr.ai FREE TIER Deployment'

Resources:
  FactrAISecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for factr.ai FREE TIER
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

  FactrAIInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c02fb55956c7d316
      InstanceType: t3.micro
      SecurityGroupIds:
        - !Ref FactrAISecurityGroup
      Tags:
        - Key: Name
          Value: factr-ai-free-instance

Outputs:
  InstancePublicIP:
    Description: Public IP of factr.ai instance
    Value: !GetAtt FactrAIInstance.PublicIpAddress
'@
    
    $template | Out-File -FilePath "aws_deployment_free.yaml" -Encoding UTF8
    Write-Host "Template created!" -ForegroundColor Green
}

# Upload template
try {
    aws s3 cp aws_deployment_free.yaml s3://$BUCKET_NAME/
    Write-Host "Template uploaded to S3!" -ForegroundColor Green
} catch {
    Write-Host "Failed to upload template!" -ForegroundColor Red
    exit 1
}

# Step 3: Deploy Infrastructure
Write-Host ""
Write-Host "Step 3: Deploying FREE TIER Infrastructure..." -ForegroundColor Yellow

$STACK_NAME = "factr-ai-free-tier"
$TEMPLATE_URL = "https://s3.amazonaws.com/$BUCKET_NAME/aws_deployment_free.yaml"

Write-Host "Stack Name: $STACK_NAME"
Write-Host "Instance Type: t3.micro (FREE)"
Write-Host "Monthly Cost: $0.00" -ForegroundColor Green

try {
    aws cloudformation create-stack --stack-name $STACK_NAME --template-url $TEMPLATE_URL
    Write-Host "Deployment started!" -ForegroundColor Green
} catch {
    Write-Host "Deployment failed!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)"
    exit 1
}

# Step 4: Monitor Deployment
Write-Host ""
Write-Host "Step 4: Monitoring deployment..." -ForegroundColor Yellow
Write-Host "This will take 5-10 minutes..."

$attempts = 0
$maxAttempts = 20

do {
    Start-Sleep -Seconds 60
    $attempts++
    
    try {
        $STATUS = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].StackStatus' --output text 2>$null
    } catch {
        $STATUS = "PENDING"
    }
    
    Write-Host "Attempt $attempts/$maxAttempts - Status: $STATUS" -ForegroundColor Yellow
    
    if ($STATUS -eq "CREATE_COMPLETE") {
        Write-Host "Deployment complete!" -ForegroundColor Green
        break
    } elseif ($STATUS -like "*FAILED*" -or $STATUS -like "*ROLLBACK*") {
        Write-Host "Deployment failed!" -ForegroundColor Red
        
        # Get failure details
        try {
            Write-Host "Getting failure details..."
            aws cloudformation describe-stack-events --stack-name $STACK_NAME --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' --output table
        } catch {}
        
        exit 1
    }
    
    if ($attempts -ge $maxAttempts) {
        Write-Host "Deployment timed out!" -ForegroundColor Red
        exit 1
    }
    
} while ($true)

# Step 5: Get Results
Write-Host ""
Write-Host "Step 5: Getting deployment info..." -ForegroundColor Yellow

try {
    $PUBLIC_IP = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`InstancePublicIP`].OutputValue' --output text
    
    if ($PUBLIC_IP) {
        Write-Host ""
        Write-Host "FREE TIER Deployment Complete!" -ForegroundColor Green
        Write-Host "================================="
        Write-Host "Monthly Cost: $0.00" -ForegroundColor Cyan
        Write-Host "Public IP: $PUBLIC_IP" -ForegroundColor Green
        Write-Host "API will be at: http://$PUBLIC_IP:8000" -ForegroundColor Green
        
        # Save deployment info
        $deploymentInfo = @{
            StackName = $STACK_NAME
            PublicIP = $PUBLIC_IP
            APIEndpoint = "http://$PUBLIC_IP:8000"
            BucketName = $BUCKET_NAME
            Region = $region
        }
        
        $deploymentInfo | ConvertTo-Json | Out-File -FilePath "deployment-free-info.json"
        Write-Host ""
        Write-Host "Deployment info saved to deployment-free-info.json"
        Write-Host ""
        Write-Host "Next Steps:"
        Write-Host "1. Wait 5-10 minutes for instance to boot"
        Write-Host "2. SSH into instance to install factr.ai"
        Write-Host "3. Test the deployment"
        
    } else {
        Write-Host "Could not get public IP!" -ForegroundColor Red
    }
    
} catch {
    Write-Host "Failed to get deployment info!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Infrastructure deployment completed!" -ForegroundColor Green
'@ | Out-File -FilePath "deploy-free-tier-clean.ps1" -Encoding UTF8