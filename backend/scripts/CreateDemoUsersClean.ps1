# CreateDemoUsersClean.ps1 - Create demo users for RecursiaDx authentication testing

$apiUrl = "http://localhost:5001/api/auth/register"

# Demo users data
$users = @(
    @{
        name = "Lab Technician"
        email = "tech@recursiadx.com" 
        password = "Demo123!"
        role = "technician"
    },
    @{
        name = "Dr. Sarah Pathologist"
        email = "pathologist@recursiadx.com"
        password = "Demo123!"
        role = "pathologist" 
    },
    @{
        name = "Admin User"
        email = "admin@recursiadx.com"
        password = "Demo123!"
        role = "admin"
    }
)

Write-Host "Creating demo users for RecursiaDx..." -ForegroundColor Yellow

foreach ($user in $users) {
    try {
        $body = @{
            name = $user.name
            email = $user.email
            password = $user.password
            role = $user.role
        } | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri $apiUrl -Method POST -Body $body -ContentType "application/json"
        Write-Host "✅ Created user: $($user.email) ($($user.role))" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to create $($user.email): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`nDemo user creation completed!" -ForegroundColor Green
Write-Host "`nDemo Login Credentials:" -ForegroundColor Cyan
Write-Host "Email: tech@recursiadx.com | Password: Demo123! (Lab Technician)" -ForegroundColor White
Write-Host "Email: pathologist@recursiadx.com | Password: Demo123! (Pathologist)" -ForegroundColor White  
Write-Host "Email: admin@recursiadx.com | Password: Demo123! (Admin)" -ForegroundColor White
Write-Host "`nYou can now test login at: http://localhost:5173" -ForegroundColor Cyan