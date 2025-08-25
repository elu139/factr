# factr.ai FREE TIER Management Script
# Manage your $0/month deployment to stay within limits

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("status", "start", "stop", "restart", "usage", "cleanup")]
    [string]$Action
)

Write-Host "🛠️ factr.ai FREE TIER Management" -ForegroundColor Green
Write-Host "================================"
Write-Host "💰 Managing your `$0.00/month deployment" -ForegroundColor Cyan

# Load deployment info
if (-not (Test-Path "deployment-free-info.json")) {
    Write-Host "❌ deployment-free-info.json not found!" -ForegroundColor Red
    Write-Host "Run deploy-free-tier.ps1 first."
    exit 1
}

$deploymentInfo = Get-Content "deployment-free-info.json" | ConvertFrom-Json

# Get instance ID
try {
    $instanceId = aws ec2 describe-instances --filters "Name=tag:Name,Values=production-factr-ai-free-instance" --query 'Reservations[0].Instances[0].InstanceId' --output text
    
    if (-not $instanceId -or $instanceId -eq "None") {
        Write-Host "❌ Could not find FREE TIER instance!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "📋 Instance ID: $instanceId" -ForegroundColor Yellow
} catch {
    Write-Host "❌ Error finding instance: $_" -ForegroundColor Red
    exit 1
}

switch ($Action) {
    "status" {
        Write-Host "`n📊 FREE TIER Status Check" -ForegroundColor Yellow
        Write-Host "========================="
        
        try {
            # Get instance state
            $instanceState = aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].State.Name' --output text
            $instanceType = aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].InstanceType' --output text
            $launchTime = aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].LaunchTime' --output text
            
            Write-Host "🖥️  Instance Status:" -ForegroundColor Cyan
            Write-Host "   ID: $instanceId"
            Write-Host "   State: $instanceState" -ForegroundColor $(if ($instanceState -eq "running") { "Green" } else { "Yellow" })
            Write-Host "   Type: $instanceType (FREE TIER eligible)"
            Write-Host "   Launch Time: $launchTime"
            
            # Calculate hours used today
            if ($launchTime) {
                $launch = [DateTime]::Parse($launchTime)
                $now = Get-Date
                $hoursRunning = [Math]::Round(($now - $launch).TotalHours, 2)
                Write-Host "   Hours Running: $hoursRunning" -ForegroundColor $(if ($hoursRunning -lt 24) { "Green" } else { "Yellow" })
            }
            
            # Check if API is responding
            if ($instanceState -eq "running") {
                Write-Host "`n🔍 API Status:" -ForegroundColor Cyan
                try {
                    $response = Invoke-WebRequest -Uri "$($deploymentInfo.APIEndpoint)/health" -UseBasicParsing -TimeoutSec 10
                    Write-Host "   API: ✅ Responding ($($response.StatusCode))" -ForegroundColor Green
                } catch {
                    Write-Host "   API: ❌ Not responding" -ForegroundColor Red
                }
            }
            
            # Show cost info
            Write-Host "`n💰 Cost Information:" -ForegroundColor Cyan
            Write-Host "   Current Cost: `$0.00/month (FREE TIER)" -ForegroundColor Green
            Write-Host "   FREE TIER Limit: 750 hours/month"
            Write-Host "   Estimated Monthly Hours: $([Math]::Round($hoursRunning * 30, 0))" -ForegroundColor $(if ($hoursRunning * 30 -lt 750) { "Green" } else { "Red" })
            
        } catch {
            Write-Host "❌ Error getting status: $_" -ForegroundColor Red
        }
    }
    
    "start" {
        Write-Host "`n🚀 Starting FREE TIER Instance" -ForegroundColor Yellow
        Write-Host "=============================="
        
        try {
            $currentState = aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].State.Name' --output text
            
            if ($currentState -eq "running") {
                Write-Host "✅ Instance is already running!" -ForegroundColor Green
            } else {
                Write-Host "Starting instance $instanceId..."
                aws ec2 start-instances --instance-ids $instanceId | Out-Null
                
                Write-Host "⏳ Waiting for instance to start..."
                aws ec2 wait instance-running --instance-ids $instanceId
                
                Write-Host "✅ Instance started successfully!" -ForegroundColor Green
                Write-Host "⏳ Wait 2-3 minutes for application to be ready"
                
                # Show connection info
                $publicIp = aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
                Write-Host "`n🔗 Connection Info:" -ForegroundColor Cyan
                Write-Host "   API: http://${publicIp}:8000"
                Write-Host "   Status: http://${publicIp}"
            }
        } catch {
            Write-Host "❌ Error starting instance: $_" -ForegroundColor Red
        }
    }
    
    "stop" {
        Write-Host "`n🛑 Stopping FREE TIER Instance" -ForegroundColor Yellow
        Write-Host "==============================="
        Write-Host "💡 This saves FREE TIER hours when not in use!" -ForegroundColor Cyan
        
        try {
            $currentState = aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].State.Name' --output text
            
            if ($currentState -eq "stopped") {
                Write-Host "✅ Instance is already stopped!" -ForegroundColor Green
            } else {
                Write-Host "Stopping instance $instanceId..."
                aws ec2 stop-instances --instance-ids $instanceId | Out-Null
                
                Write-Host "⏳ Waiting for instance to stop..."
                aws ec2 wait instance-stopped --instance-ids $instanceId
                
                Write-Host "✅ Instance stopped successfully!" -ForegroundColor Green
                Write-Host "💰 FREE TIER hours saved!" -ForegroundColor Cyan
                
                # Calculate hours saved
                $now = Get-Date
                $hoursInMonth = [DateTime]::DaysInMonth($now.Year, $now.Month) * 24
                $remainingHours = $hoursInMonth - ($now.Day * 24)
                Write-Host "💡 Potential hours saved this month: $remainingHours" -ForegroundColor Green
            }
        } catch {
            Write-Host "❌ Error stopping instance: $_" -ForegroundColor Red
        }
    }
    
    "restart" {
        Write-Host "`n🔄 Restarting FREE TIER Instance" -ForegroundColor Yellow
        Write-Host "================================"
        
        try {
            Write-Host "Restarting instance $instanceId..."
            aws ec2 reboot-instances --instance-ids $instanceId
            
            Write-Host "⏳ Waiting for restart to complete..."
            Start-Sleep -Seconds 30
            
            # Wait for instance to be running again
            aws ec2 wait instance-running --instance-ids $instanceId
            
            Write-Host "✅ Instance restarted successfully!" -ForegroundColor Green
            Write-Host "⏳ Wait 2-3 minutes for application to be ready"
        } catch {
            Write-Host "❌ Error restarting instance: $_" -ForegroundColor Red
        }
    }
    
    "usage" {
        Write-Host "`n📊 FREE TIER Usage Analysis" -ForegroundColor Yellow
        Write-Host "============================"
        
        try {
            # Get current month info
            $now = Get-Date
            $monthStart = Get-Date -Day 1 -Hour 0 -Minute 0 -Second 0
            $daysInMonth = [DateTime]::DaysInMonth($now.Year, $now.Month)
            $daysPassed = $now.Day
            $daysRemaining = $daysInMonth - $daysPassed
            
            Write-Host "📅 Current Period:" -ForegroundColor Cyan
            Write-Host "   Month: $($now.ToString('MMMM yyyy'))"
            Write-Host "   Days Passed: $daysPassed / $daysInMonth"
            Write-Host "   Days Remaining: $daysRemaining"
            
            # Estimate usage
            $maxHoursThisMonth = $daysInMonth * 24
            $hoursPassedIfAlwaysOn = $daysPassed * 24
            $hoursRemainingIfAlwaysOn = $daysRemaining * 24
            
            Write-Host "`n⏰ FREE TIER Hours:" -ForegroundColor Cyan
            Write-Host "   FREE TIER Limit: 750 hours/month" -ForegroundColor Green
            Write-Host "   Max Possible Hours: $maxHoursThisMonth hours/month"
            Write-Host "   Hours if always on: $hoursPassedIfAlwaysOn (so far)"
            
            if ($hoursPassedIfAlwaysOn -gt 750) {
                Write-Host "   ⚠️ Would exceed FREE TIER if always on!" -ForegroundColor Red
                Write-Host "   💡 Recommended: Stop instance when not in use" -ForegroundColor Yellow
            } else {
                Write-Host "   ✅ Within FREE TIER limits if always on" -ForegroundColor Green
            }
            
            # Recommendations
            Write-Host "`n💡 Usage Optimization:" -ForegroundColor Yellow
            Write-Host "   • Run 24/7: Uses $maxHoursThisMonth hours" -ForegroundColor $(if ($maxHoursThisMonth -gt 750) { "Red" } else { "Green" })
            Write-Host "   • Run 8h/day: Uses $($daysInMonth * 8) hours" -ForegroundColor Green
            Write-Host "   • Run 12h/day: Uses $($daysInMonth * 12) hours" -ForegroundColor Green
            Write-Host "   • Business hours only: Uses ~$($daysInMonth * 8) hours" -ForegroundColor Green
            
            Write-Host "`n📊 Monitoring:" -ForegroundColor Cyan
            Write-Host "   • AWS Billing Dashboard: https://console.aws.amazon.com/billing/home#/freetier"
            Write-Host "   • Set up billing alerts for safety"
            Write-Host "   • Check usage weekly"
            
        } catch {
            Write-Host "❌ Error analyzing usage: $_" -ForegroundColor Red
        }
    }
    
    "cleanup" {
        Write-Host "`n🧹 FREE TIER Cleanup" -ForegroundColor Yellow
        Write-Host "===================="
        Write-Host "⚠️ This will DELETE your FREE TIER deployment!" -ForegroundColor Red
        
        $confirm = Read-Host "Are you sure you want to delete everything? (type 'DELETE' to confirm)"
        
        if ($confirm -eq "DELETE") {
            try {
                Write-Host "`nDeleting CloudFormation stack..."
                aws cloudformation delete-stack --stack-name $deploymentInfo.StackName
                
                Write-Host "⏳ Waiting for stack deletion..."
                aws cloudformation wait stack-delete-complete --stack-name $deploymentInfo.StackName
                
                Write-Host "✅ CloudFormation stack deleted!" -ForegroundColor Green
                
                # Clean up deployment bucket
                Write-Host "`nCleaning up deployment bucket..."
                aws s3 rb s3://$($deploymentInfo.BucketName) --force
                
                Write-Host "✅ Deployment bucket cleaned up!" -ForegroundColor Green
                
                # Remove local files
                if (Test-Path "deployment-free-info.json") {
                    Remove-Item "deployment-free-info.json"
                    Write-Host "✅ Local deployment info cleaned up!" -ForegroundColor Green
                }
                
                Write-Host "`n🎉 FREE TIER cleanup completed!" -ForegroundColor Green
                Write-Host "💰 Your AWS account is now clean and ready for other projects" -ForegroundColor Cyan
                
            } catch {
                Write-Host "❌ Error during cleanup: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "Cleanup cancelled." -ForegroundColor Yellow
        }
    }
}

Write-Host "`n📖 Available Commands:" -ForegroundColor Cyan
Write-Host "   .\manage-free-tier.ps1 status    - Check instance status"
Write-Host "   .\manage-free-tier.ps1 start     - Start the instance"
Write-Host "   .\manage-free-tier.ps1 stop      - Stop the instance (save hours!)"
Write-Host "   .\manage-free-tier.ps1 restart   - Restart the instance"
Write-Host "   .\manage-free-tier.ps1 usage     - Analyze FREE TIER usage"
Write-Host "   .\manage-free-tier.ps1 cleanup   - Delete everything"

Write-Host "`n💰 Remember: Stop instance when not in use to maximize FREE TIER!" -ForegroundColor Green