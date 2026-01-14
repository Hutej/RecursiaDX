import fetch from 'node-fetch';

const API_BASE_URL = 'http://localhost:5001/api';

const demoUsers = [
  {
    name: 'Tech User',
    email: 'tech@recursiadx.com',
    password: 'Demo123!',
    role: 'Lab Technician',
    department: 'Clinical Laboratory',
    phone: '+1-555-0101'
  },
  {
    name: 'Dr. Smith',
    email: 'pathologist@recursiadx.com',
    password: 'Demo123!',
    role: 'Pathologist',
    department: 'Pathology Department',
    licenseNumber: 'PATH12345',
    phone: '+1-555-0102'
  },
  {
    name: 'Admin User',
    email: 'admin@recursiadx.com',
    password: 'Demo123!',
    role: 'Admin',
    department: 'Administration',
    phone: '+1-555-0103'
  }
];

const createDemoUsers = async () => {
  console.log('🚀 Creating demo users...\n');

  for (const user of demoUsers) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(user),
      });

      const result = await response.json();

      if (response.ok) {
        console.log(`✅ Created user: ${user.email} (${user.role})`);
      } else {
        if (result.message && result.message.includes('already exists')) {
          console.log(`ℹ️  User already exists: ${user.email}`);
        } else {
          console.log(`❌ Failed to create ${user.email}: ${result.message}`);
        }
      }
    } catch (error) {
      console.log(`❌ Error creating ${user.email}: ${error.message}`);
    }
  }

  console.log('\n✨ Demo user creation completed!');
  console.log('\nDemo Login Credentials:');
  console.log('📧 tech@recursiadx.com | 🔐 Demo123! (Lab Technician)');
  console.log('📧 pathologist@recursiadx.com | 🔐 Demo123! (Pathologist)');
  console.log('📧 admin@recursiadx.com | 🔐 Demo123! (Admin)');
  console.log('\nYou can now test login at: http://localhost:5173');
};

createDemoUsers().catch(console.error);