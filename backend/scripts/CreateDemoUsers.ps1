# Create demo users for RecursiaDx

$apiBaseUrl = "http://localhost:5001/api"

$demoUsers = @(
    @{
        name = "Tech User"
        email = "tech@recursiadx.com"
        password = "Demo123!"
        role = "Lab Technician"
        department = "Clinical Laboratory"
        phone = "+1-555-0101"
    },
    @{
        name = "Dr. Smith"
        email = "pathologist@recursiadx.com"
        password = "Demo123!"
        role = "Pathologist"
        department = "Pathology Department"
        licenseNumber = "PATH12345"
        phone = "+1-555-0102"
    },
    @{
        name = "Admin User"
        email = "admin@recursiadx.com"
        password = "Demo123!"
        role = "Admin"
        department = "Administration"
        phone = "+1-555-0103"
    }
)

Write-Host "🚀 Creating demo users...`n" -ForegroundColor Green

foreach ($user in $demoUsers) {
    try {
        $body = $user | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$apiBaseUrl/auth/register" -Method POST -ContentType "application/json" -Body $body
        
        if ($response.success) {
            Write-Host "✅ Created user: $($user.email) ($($user.role))" -ForegroundColor Green
        }
    }
    catch {
        $errorResponse = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($errorResponse.message -and $errorResponse.message -match "already exists") {
            Write-Host "ℹ️  User already exists: $($user.email)" -ForegroundColor Yellow
        }
        else {
            Write-Host "❌ Failed to create $($user.email): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

Write-Host "`n✨ Demo user creation completed!" -ForegroundColor Green
Write-Host "`nDemo Login Credentials:" -ForegroundColor Cyan
Write-Host "📧 tech@recursiadx.com | 🔐 Demo123! (Lab Technician)" -ForegroundColor White
Write-Host "📧 pathologist@recursiadx.com | 🔐 Demo123! (Pathologist)" -ForegroundColor White  
Write-Host "📧 admin@recursiadx.com | 🔐 Demo123! (Admin)" -ForegroundColor White
Write-Host "`nYou can now test login at: http://localhost:5173" -ForegroundColor Cyan