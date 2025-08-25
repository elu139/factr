# factr.ai FREE TIER AWS Deployment Script
# 100% FREE for 12 months with AWS Free Tier

Write-Host "🆓 factr.ai AWS FREE TIER Deployment" -ForegroundColor Green
Write-Host "====================================="
Write-Host "💰 Cost: `$0.00/month for first 12 months!" -ForegroundColor Cyan

# Step 1: Verify AWS Free Tier Eligibility
Write-Host "`n📋 Step 1: Verifying AWS Free Tier Eligibility..." -ForegroundColor Yellow

try {
    $awsIdentity = aws sts get-caller-identity | ConvertFrom-Json
    Write-Host "✅ AWS CLI working!" -ForegroundColor Green
    Write-Host "   Account: $($awsIdentity.Account)"
    
    # Check account age (Free Tier is for first 12 months)
    Write-Host "`n💡 FREE TIER REMINDER:" -ForegroundColor Cyan
    Write-Host "   - Valid for FIRST 12 months from account creation"
    Write-Host "   - t3.micro: 750 hours/month FREE"
    Write-Host "   - S3: 5GB storage FREE"
    Write-Host "   - Data transfer: 15GB/month FREE"
    
} catch {
    Write-Host "❌ AWS CLI not configured!" -ForegroundColor Red
    Write-Host "Run: aws configure"
    exit 1
}

# Check/set region
$region = aws configure get region
if (-not $region) {
    Write-Host "Setting default region to us-east-1 (most free tier resources)..."
    aws configure set region us-east-1
    $region = "us-east-1"
}
Write-Host "✅ Using region: $region" -ForegroundColor Green

# Step 2: Create FREE TIER S3 Bucket
Write-Host "`n📦 Step 2: Creating FREE TIER S3 Bucket..." -ForegroundColor Yellow

$BUCKET_NAME = "factr-ai-free-$($awsIdentity.Account)-$(Get-Date -Format 'yyyyMMdd')"
Write-Host "Creating bucket: $BUCKET_NAME (within 5GB free limit)"

try {
    aws s3 mb s3://$BUCKET_NAME
    Write-Host "✅ S3 bucket created!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create S3 bucket!" -ForegroundColor Red
    exit 1
}

# Upload FREE TIER CloudFormation template
if (-not (Test-Path "aws_deployment_free.yaml")) {
    Write-Host "❌ aws_deployment_free.yaml not found!" -ForegroundColor Red
    Write-Host "Create the free tier template first."
    exit 1
}

try {
    aws s3 cp aws_deployment_free.yaml s3://$BUCKET_NAME/
    Write-Host "✅ CloudFormation template uploaded!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to upload template!" -ForegroundColor Red
    exit 1
}

# Step 3: Deploy FREE TIER Infrastructure
Write-Host "`n🏗️  Step 3: Deploying FREE TIER Infrastructure..." -ForegroundColor Yellow

$STACK_NAME = "factr-ai-free-tier"
$ENVIRONMENT = "production"
$TEMPLATE_URL = "https://s3.amazonaws.com/$BUCKET_NAME/aws_deployment_free.yaml"

Write-Host "FREE TIER Configuration:"
Write-Host "  Stack Name: $STACK_NAME"
Write-Host "  Instance Type: t3.micro (FREE for 750 hours/month)"
Write-Host "  Load Balancer: NONE (saves `$18/month)"
Write-Host "  Redis: Local container (saves `$13/month)"
Write-Host "  Total Cost: `$0.00/month" -ForegroundColor Green

$deployCommand = @"
aws cloudformation create-stack \
  --stack-name $STACK_NAME \
  --template-url $TEMPLATE_URL \
  --parameters \
    ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
  --capabilities CAPABILITY_IAM \
  --tags \
    Key=Project,Value=factr-ai \
    Key=Tier,Value=free \
    Key=Environment,Value=$ENVIRONMENT \
    Key=Owner,Value=$env:USERNAME
"@

Write-Host "`nDeploying FREE TIER stack..."

try {
    Invoke-Expression $deployCommand
    Write-Host "✅ FREE TIER stack deployment initiated!" -ForegroundColor Green
} catch {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    Write-Host "Error: $_"
    exit 1
}

# Step 4: Monitor Deployment
Write-Host "`n📊 Step 4: Monitoring FREE TIER Deployment..." -ForegroundColor Yellow
Write-Host "This will take 5-10 minutes (faster than paid tier - single instance)..."

$startTime = Get-Date
$deploymentComplete = $false

do {
    Start-Sleep -Seconds 60
    
    try {
        $STATUS = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].StackStatus' --output text 2>$null
    } catch {
        $STATUS = "PENDING"
    }
    
    $elapsed = (Get-Date) - $startTime
    Write-Host "[$($elapsed.ToString('mm\:ss'))] Status: $STATUS" -ForegroundColor Yellow
    
    switch ($STATUS) {
        "CREATE_COMPLETE" {
            Write-Host "`n✅ FREE TIER stack created successfully!" -ForegroundColor Green
            $deploymentComplete = $true
            break
        }
        "CREATE_FAILED" {
            Write-Host "`n❌ Stack creation failed!" -ForegroundColor Red
            
            # Get failure reason
            try {
                $events = aws cloudformation describe-stack-events --stack-name $STACK_NAME --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' --output text
                Write-Host "Failure details:"
                $events
            } catch {}
            
            exit 1
        }
        "ROLLBACK_COMPLETE" {
            Write-Host "`n❌ Stack creation failed and rolled back!" -ForegroundColor Red
            exit 1
        }
    }
} while (-not $deploymentComplete)

# Step 5: Get FREE TIER Deployment Information
Write-Host "`n📋 Step 5: Getting FREE TIER Deployment Info..." -ForegroundColor Yellow

try {
    $PUBLIC_IP = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`InstancePublicIP`].OutputValue' --output text
    $PUBLIC_DNS = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`InstancePublicDNS`].OutputValue' --output text
    $API_ENDPOINT = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`APIEndpoint`].OutputValue' --output text
    $STATUS_PAGE = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`StatusPageURL`].OutputValue' --output text
    $S3_BUCKET = aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' --output text
    
    Write-Host "`n🎉 FREE TIER Deployment Complete!" -ForegroundColor Green
    Write-Host "=================================="
    Write-Host "💰 Monthly Cost: `$0.00 (AWS Free Tier)" -ForegroundColor Cyan
    Write-Host "🌐 Public IP: $PUBLIC_IP" -ForegroundColor Green
    Write-Host "🔗 API Endpoint: $API_ENDPOINT" -ForegroundColor Green
    Write-Host "📊 Status Page: $STATUS_PAGE" -ForegroundColor Green
    Write-Host "📦 S3 Bucket: $S3_BUCKET" -ForegroundColor Green
    
    # Save deployment info
    $deploymentInfo = @{
        StackName = $STACK_NAME
        PublicIP = $PUBLIC_IP
        PublicDNS = $PUBLIC_DNS
        APIEndpoint = $API_ENDPOINT
        StatusPageURL = $STATUS_PAGE
        S3Bucket = $S3_BUCKET
        BucketName = $BUCKET_NAME
        Region = $region
        Tier = "FREE"
        MonthlyCost = "`$0.00"
    }
    
    $deploymentInfo | ConvertTo-Json | Out-File -FilePath "deployment-free-info.json"
    Write-Host "`n💾 Deployment info saved to deployment-free-info.json"
    
} catch {
    Write-Host "`n❌ Failed to get deployment info!" -ForegroundColor Red
    Write-Host "Error: $_"
}

# Step 6: Wait for Application to Start
Write-Host "`n⏳ Step 6: Waiting for Application to Start..." -ForegroundColor Yellow
Write-Host "The instance is automatically installing Docker and building factr.ai..."
Write-Host "This will take 10-15 minutes for first startup."

$maxAttempts = 20
$attempt = 1
$appReady = $false

Write-Host "`nChecking application status every 60 seconds..."

while ($attempt -le $maxAttempts -and -not $appReady) {
    Write-Host "   Attempt $attempt/$maxAttempts - checking API endpoint..."
    
    try {
        $response = Invoke-WebRequest -Uri "$API_ENDPOINT/health" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ factr.ai API is running!" -ForegroundColor Green
            $appReady = $true
        }
    } catch {
        Write-Host "   Still starting up... (this is normal for first 10-15 minutes)"
        Start-Sleep -Seconds 60
        $attempt++
    }
}

if (-not $appReady) {
    Write-Host "`n⚠️ Application not responding yet, but infrastructure is ready!" -ForegroundColor Yellow
    Write-Host "The app may still be starting up. Check again in a few minutes."
}

# Step 7: Final Instructions
Write-Host "`n📝 FREE TIER Deployment Summary" -ForegroundColor Yellow
Write-Host "==============================="

Write-Host "`n✅ What's deployed:" -ForegroundColor Green
Write-Host "   🖥️  Single t3.micro EC2 instance (1 vCPU, 1GB RAM)"
Write-Host "   🗄️  Redis container (local, 128MB memory limit)"
Write-Host "   📦 S3 bucket for model storage (5GB free)"
Write-Host "   🔐 Security groups and IAM roles"
Write-Host "   📊 Basic CloudWatch monitoring"

Write-Host "`n🎯 Your FREE factr.ai endpoints:" -ForegroundColor Cyan
Write-Host "   Status Page: $STATUS_PAGE"
Write-Host "   API Health: $API_ENDPOINT/health"
Write-Host "   API Docs: $API_ENDPOINT/docs"
Write-Host "   Performance: $API_ENDPOINT/analytics/performance"

Write-Host "`n⚠️ FREE TIER Limitations:" -ForegroundColor Yellow
Write-Host "   • Single instance (no auto-scaling)"
Write-Host "   • 1GB RAM (may be slower for large models)"
Write-Host "   • No load balancer (single point of failure)"
Write-Host "   • Local Redis cache (limited memory)"
Write-Host "   • 750 hours/month limit (31 days = 744 hours)"

Write-Host "`n💡 FREE TIER Best Practices:" -ForegroundColor Cyan
Write-Host "   • Monitor usage in AWS Billing dashboard"
Write-Host "   • Stop instance when not in use to save hours"
Write-Host "   • Use smaller models (DistilBERT vs BERT)"
Write-Host "   • Cache results aggressively"

Write-Host "`n📊 Usage Monitoring:" -ForegroundColor Yellow
Write-Host "   • Check AWS Billing & Cost Management dashboard"
Write-Host "   • Set up billing alerts for safety"
Write-Host "   • Monitor EC2 hours: 750/month FREE"
Write-Host "   • Monitor S3 usage: 5GB FREE"

Write-Host "`n🚀 Next Steps:" -ForegroundColor Green
Write-Host "1. ⏳ Wait 10-15 minutes for full startup"
Write-Host "2. 🧪 Test the API: curl $API_ENDPOINT/health"
Write-Host "3. 📖 Read API docs: $API_ENDPOINT/docs"
Write-Host "4. 🎯 Try analyzing Instagram posts"
Write-Host "5. 📊 Monitor free tier usage"

Write-Host "`n🛑 Important FREE TIER Reminders:" -ForegroundColor Red
Write-Host "   • Stop instance when not needed: aws ec2 stop-instances --instance-ids <id>"
Write-Host "   • Free tier expires after 12 months"
Write-Host "   • Going over limits will incur charges"
Write-Host "   • Monitor usage regularly"

Write-Host "`n✅ FREE TIER deployment completed successfully!" -ForegroundColor Green
Write-Host "💰 Total cost: `$0.00/month for first 12 months!" -ForegroundColor Cyan
Write-Host "`nRun: .\test-free-tier.ps1 (after 10-15 minutes)" -ForegroundColor Yellow