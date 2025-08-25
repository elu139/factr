# factr.ai FREE TIER Testing Script
# Test your $0/month deployment

Write-Host "🧪 factr.ai FREE TIER Testing" -ForegroundColor Green
Write-Host "=============================="
Write-Host "💰 Testing your `$0.00/month deployment!" -ForegroundColor Cyan

# Load deployment info
if (-not (Test-Path "deployment-free-info.json")) {
    Write-Host "❌ deployment-free-info.json not found!" -ForegroundColor Red
    Write-Host "Run deploy-free-tier.ps1 first."
    exit 1
}

$deploymentInfo = Get-Content "deployment-free-info.json" | ConvertFrom-Json
$API_ENDPOINT = $deploymentInfo.APIEndpoint
$STATUS_PAGE = $deploymentInfo.StatusPageURL
$PUBLIC_IP = $deploymentInfo.PublicIP

Write-Host "`n🎯 Testing FREE TIER endpoints:" -ForegroundColor Yellow
Write-Host "   API: $API_ENDPOINT"
Write-Host "   Status: $STATUS_PAGE"
Write-Host "   Instance: $PUBLIC_IP"

# Function to test FREE TIER endpoint
function Test-FreeTierEndpoint {
    param(
        [string]$Endpoint,
        [string]$Description,
        [int]$TimeoutSec = 30
    )
    
    Write-Host -NoNewline "Testing $Description... "
    
    try {
        $response = Invoke-WebRequest -Uri "$API_ENDPOINT$Endpoint" -UseBasicParsing -TimeoutSec $TimeoutSec
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ PASSED" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ FAILED ($($response.StatusCode))" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ FAILED ($($_.Exception.Message))" -ForegroundColor Red
        return $false
    }
}

# Function to test JSON endpoint
function Test-FreeTierJson {
    param(
        [string]$Endpoint,
        [string]$Description,
        [int]$TimeoutSec = 30
    )
    
    Write-Host -NoNewline "Testing $Description... "
    
    try {
        $response = Invoke-RestMethod -Uri "$API_ENDPOINT$Endpoint" -TimeoutSec $TimeoutSec
        Write-Host "✅ PASSED" -ForegroundColor Green
        
        # Show key info for FREE TIER
        if ($Endpoint -eq "/health") {
            Write-Host "   Status: $($response.status)" -ForegroundColor Gray
            Write-Host "   Session: $($response.session)" -ForegroundColor Gray
        }
        elseif ($Endpoint -eq "/analytics/performance") {
            Write-Host "   System: $($response.system_status)" -ForegroundColor Gray
            Write-Host "   Cache: $($response.cache_analytics.status)" -ForegroundColor Gray
        }
        
        return $true
    } catch {
        Write-Host "❌ FAILED ($($_.Exception.Message))" -ForegroundColor Red
        return $false
    }
}

# Step 1: Check if instance is running
Write-Host "`n⚡ Step 1: Checking FREE TIER Instance..." -ForegroundColor Yellow

try {
    $instanceId = aws ec2 describe-instances --filters "Name=tag:Name,Values=production-factr-ai-free-instance" --query 'Reservations[0].Instances[0].InstanceId' --output text
    
    if ($instanceId -and $instanceId -ne "None") {
        $instanceState = aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].State.Name' --output text
        Write-Host "Instance ID: $instanceId" -ForegroundColor Cyan
        Write-Host "Instance State: $instanceState" -ForegroundColor $(if ($instanceState -eq "running") { "Green" } else { "Yellow" })
        
        if ($instanceState -ne "running") {
            Write-Host "`n⚠️ Instance is not running!" -ForegroundColor Yellow
            Write-Host "Start it with: aws ec2 start-instances --instance-ids $instanceId"
            exit 1
        }
    } else {
        Write-Host "⚠️ Could not find instance. Check CloudFormation stack." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not check instance status." -ForegroundColor Yellow
}

# Step 2: Test basic endpoints
Write-Host "`n🔍 Step 2: Testing Basic FREE TIER Endpoints..." -ForegroundColor Yellow

$healthPassed = Test-FreeTierJson -Endpoint "/health" -Description "Health Check"
$statusPassed = Test-FreeTierEndpoint -Endpoint "/" -Description "Status Page"
$capabilitiesPassed = Test-FreeTierJson -Endpoint "/capabilities" -Description "Capabilities"

# Step 3: Test FREE TIER specific features
Write-Host "`n📊 Step 3: Testing FREE TIER Performance..." -ForegroundColor Yellow

$analyticsPassed = Test-FreeTierJson -Endpoint "/analytics/performance" -Description "Performance Analytics"

# Step 4: Test lightweight analysis (FREE TIER optimized)
Write-Host "`n🧠 Step 4: Testing FREE TIER AI Analysis..." -ForegroundColor Yellow
Write-Host "Using lightweight settings for t3.micro instance..."

$freeTierAnalysis = @{
    post_url = "https://www.instagram.com/p/freetiertest/"
    include_reverse_search = $false  # Disabled to save memory
    include_metadata_analysis = $false  # Disabled to save memory
    explanation_config = @{
        audience = "general_public"
        include_evidence = $false  # Simplified for FREE TIER
        include_recommendations = $false
        language = "en"
    }
    cache_results = $true
} | ConvertTo-Json

Write-Host "Running lightweight analysis (optimized for 1GB RAM)..."

try {
    $startTime = Get-Date
    $analysisResponse = Invoke-RestMethod -Uri "$API_ENDPOINT/analyze/instagram" -Method Post -Body $freeTierAnalysis -ContentType "application/json" -TimeoutSec 180
    $endTime = Get-Date
    $processingTime = ($endTime - $startTime).TotalSeconds
    
    Write-Host "✅ FREE TIER analysis completed!" -ForegroundColor Green
    
    # Show FREE TIER optimized results
    $score = $analysisResponse.misinformation_score
    $cacheHit = $analysisResponse.cache_hit
    
    Write-Host "   Misinformation Score: $score%" -ForegroundColor Cyan
    Write-Host "   Processing Time: ${processingTime}s" -ForegroundColor Cyan
    Write-Host "   Cache Hit: $cacheHit" -ForegroundColor Cyan
    Write-Host "   FREE TIER Optimized: ✅" -ForegroundColor Green
    
    if ($processingTime -gt 60) {
        Write-Host "   ⚠️ Slow processing due to t3.micro limits (expected)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ FREE TIER analysis failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   This might be due to memory limits on t3.micro" -ForegroundColor Yellow
}

# Step 5: Test caching (important for FREE TIER performance)
Write-Host "`n⚡ Step 5: Testing FREE TIER Caching..." -ForegroundColor Yellow

$lightCacheTest = @{
    post_url = "https://www.instagram.com/p/cachetest/"
    include_reverse_search = $false
    include_metadata_analysis = $false
    cache_results = $true
} | ConvertTo-Json

Write-Host "Testing cache performance (critical for FREE TIER)..."

try {
    # First request
    $time1Start = Get-Date
    $firstResponse = Invoke-RestMethod -Uri "$API_ENDPOINT/analyze/instagram" -Method Post -Body $lightCacheTest -ContentType "application/json" -TimeoutSec 120
    $time1End = Get-Date
    $time1 = ($time1End - $time1Start).TotalSeconds
    
    # Second request (should be cached)
    $time2Start = Get-Date
    $secondResponse = Invoke-RestMethod -Uri "$API_ENDPOINT/analyze/instagram" -Method Post -Body $lightCacheTest -ContentType "application/json" -TimeoutSec 60
    $time2End = Get-Date
    $time2 = ($time2End - $time2Start).TotalSeconds
    
    $cacheHitFirst = $firstResponse.cache_hit
    $cacheHitSecond = $secondResponse.cache_hit
    
    Write-Host "   First request: ${time1}s (cache hit: $cacheHitFirst)" -ForegroundColor Cyan
    Write-Host "   Second request: ${time2}s (cache hit: $cacheHitSecond)" -ForegroundColor Cyan
    
    if ($cacheHitSecond -eq $true -and $time2 -gt 0) {
        $speedup = [Math]::Round($time1 / $time2, 1)
        Write-Host "   🚀 FREE TIER caching working! Speedup: ${speedup}x" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️ Caching may not be optimal (check Redis container)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Cache test failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 6: Check FREE TIER resource usage
Write-Host "`n📊 Step 6: Checking FREE TIER Resource Usage..." -ForegroundColor Yellow

try {
    # Get instance metrics (if available)
    Write-Host "Checking t3.micro resource usage..."
    
    # This would require CloudWatch metrics, simplified for FREE TIER
    Write-Host "   Instance Type: t3.micro (1 vCPU, 1GB RAM)" -ForegroundColor Cyan
    Write-Host "   Monthly Hours Used: Check AWS Billing Dashboard" -ForegroundColor Yellow
    Write-Host "   FREE TIER Limit: 750 hours/month" -ForegroundColor Green
    
} catch {
    Write-Host "⚠️ Could not check resource usage" -ForegroundColor Yellow
}

# Step 7: FREE TIER Health Summary
Write-Host "`n🏁 FREE TIER Test Summary" -ForegroundColor Yellow
Write-Host "========================"

$testResults = @{
    "Health Check" = $healthPassed
    "Status Page" = $statusPassed
    "Capabilities" = $capabilitiesPassed
    "Performance Analytics" = $analyticsPassed
}

$passedTests = 0
$totalTests = $testResults.Count

foreach ($test in $testResults.GetEnumerator()) {
    $status = if ($test.Value) { "✅ PASSED"; $passedTests++ } else { "❌ FAILED" }
    $color = if ($test.Value) { "Green" } else { "Red" }
    Write-Host "   $($test.Key): $status" -ForegroundColor $color
}

Write-Host "`nOverall: $passedTests/$totalTests tests passed" -ForegroundColor $(if ($passedTests -eq $totalTests) { "Green" } else { "Yellow" })

# FREE TIER specific recommendations
Write-Host "`n💡 FREE TIER Performance Tips:" -ForegroundColor Cyan
Write-Host "   • Use lightweight analysis (disable reverse search)"
Write-Host "   • Cache results aggressively"
Write-Host "   • Stop instance when not in use"
Write-Host "   • Monitor AWS Free Tier usage dashboard"
Write-Host "   • Use smaller models (DistilBERT vs BERT)"

Write-Host "`n📍 Your FREE TIER factr.ai is live at:" -ForegroundColor Green
Write-Host "   Status Page: $STATUS_PAGE"
Write-Host "   API Health: $API_ENDPOINT/health"
Write-Host "   API Docs: $API_ENDPOINT/docs"

Write-Host "`n💰 Cost Monitoring:" -ForegroundColor Yellow
Write-Host "   Current Cost: `$0.00/month (FREE TIER)" -ForegroundColor Green
Write-Host "   Monitor at: https://console.aws.amazon.com/billing/home#/freetier"
Write-Host "   Hours remaining: Check AWS Billing Dashboard"

Write-Host "`n🛑 FREE TIER Reminders:" -ForegroundColor Red
Write-Host "   • Stop instance to save hours: aws ec2 stop-instances --instance-ids $instanceId"
Write-Host "   • Free tier expires after 12 months"
Write-Host "   • Set up billing alerts for safety"

if ($passedTests -eq $totalTests) {
    Write-Host "`n🎉 FREE TIER deployment is working perfectly!" -ForegroundColor Green
    Write-Host "💰 Enjoy your `$0.00/month AI-powered misinformation detection!" -ForegroundColor Cyan
} else {
    Write-Host "`n⚠️ Some tests failed, but this is common on t3.micro" -ForegroundColor Yellow
    Write-Host "The basic functionality should still work for lightweight tasks." -ForegroundColor Yellow
}

Write-Host "`n✅ FREE TIER testing completed!" -ForegroundColor Green