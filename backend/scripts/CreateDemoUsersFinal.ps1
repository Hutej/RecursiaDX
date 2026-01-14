# CreateDemoUsersFinal.ps1 - Create demo users for RecursiaDx authentication testing

$apiUrl = "http://localhost:5001/api/auth/register"

# Demo users data with correct roles and confirmPassword
$users = @(
    @{
        name = "Lab Technician"
        email = "tech@recursiadx.com" 
        password = "Demo123!"
        confirmPassword = "Demo123!"
        role = "Lab Technician"
    },
    @{
        name = "Dr. Sarah Pathologist"
        email = "pathologist@recursiadx.com"
        password = "Demo123!"
        confirmPassword = "Demo123!"
        role = "Pathologist" 
    },
    @{
        name = "Admin User"
        email = "admin@recursiadx.com"
        password = "Demo123!"
        confirmPassword = "Demo123!"
        role = "Admin"
    }
)

Write-Host "Creating demo users for RecursiaDx..." -ForegroundColor Yellow

foreach ($user in $users) {
    try {
        $body = @{
            name = $user.name
            email = $user.email
            password = $user.password
            confirmPassword = $user.confirmPassword
            role = $user.role
        } | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri $apiUrl -Method POST -Body $body -ContentType "application/json"
        Write-Host "✅ Created user: $($user.email) ($($user.role))" -ForegroundColor Green
    }
    catch {
        $errorMessage = $_.Exception.Message
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Host "❌ Failed to create $($user.email): $responseBody" -ForegroundColor Red
        } else {
            Write-Host "❌ Failed to create $($user.email): $errorMessage" -ForegroundColor Red
        }
    }
}

Write-Host "`nDemo user creation completed!" -ForegroundColor Green
Write-Host "`nDemo Login Credentials:" -ForegroundColor Cyan
Write-Host "Email: tech@recursiadx.com | Password: Demo123! (Lab Technician)" -ForegroundColor White
Write-Host "Email: pathologist@recursiadx.com | Password: Demo123! (Pathologist)" -ForegroundColor White  
Write-Host "Email: admin@recursiadx.com | Password: Demo123! (Admin)" -ForegroundColor White
Write-Host "`nYou can now test login at: http://localhost:5173" -ForegroundColor Cyan