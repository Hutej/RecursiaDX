# RecursiaDx Backend API

A comprehensive Node.js backend for the RecursiaDx digital pathology platform, providing robust APIs for sample management, AI-powered analysis, and pathology workflow automation.

## Features

### 🔐 Authentication & Authorization
- JWT-based authentication with refresh tokens
- Role-based access control (Pathologist, Lab Technician, Resident, Admin, Researcher)
- Account lockout protection and rate limiting
- Password strength validation and secure hashing
- Forgot password and reset functionality

### 🧬 Sample Management
- Complete pathology sample lifecycle tracking
- Patient information management with privacy controls
- Workflow status tracking (Received → Processing → Complete)
- Image upload and metadata management
- Quality control and validation checkpoints

### 🤖 AI Analysis Integration
- AI model integration for pathology analysis
- Confidence scoring and validation workflows
- Multiple analysis types (H&E, IHC, ISH, Molecular, etc.)
- Result validation and review processes
- Performance metrics and accuracy tracking

### 📊 Reporting System
- Comprehensive pathology report generation
- Template-based report formatting
- Digital signatures and approval workflows
- Report versioning and amendment tracking
- Distribution and delivery management

### 🛡️ Security & Compliance
- HIPAA-compliant data handling
- Comprehensive audit trails
- Data encryption at rest and in transit
- Role-based data access controls
- Security headers and CORS protection

## Technology Stack

- **Runtime**: Node.js 18+
- **Framework**: Express.js
- **Database**: MongoDB with Mongoose ODM
- **Authentication**: JSON Web Tokens (JWT)
- **Validation**: Express-validator
- **Security**: Helmet, CORS, Rate limiting
- **File Processing**: Multer for uploads
- **Documentation**: Swagger/OpenAPI

## Project Structure

```
backend/
├── config/
│   └── config.js           # Environment and application configuration
├── middleware/
│   ├── auth.js            # Authentication middleware
│   ├── validation.js      # Input validation middleware
│   └── errorHandler.js    # Global error handling
├── models/
│   ├── User.js           # User model with authentication
│   ├── Sample.js         # Pathology sample model
│   ├── Analysis.js       # AI analysis results model
│   └── Report.js         # Pathology report model
├── routes/
│   ├── auth.js           # Authentication routes
│   ├── samples.js        # Sample management routes
│   ├── analysis.js       # Analysis management routes
│   └── reports.js        # Report generation routes
├── utils/
│   └── helpers.js        # Utility functions and helpers
├── server.js             # Main server file
├── package.json          # Dependencies and scripts
└── .env.example          # Environment variables template
```

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- MongoDB 4.4+
- Git

### Installation

1. **Clone and setup**
   ```bash
   cd backend
   npm install
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Required Environment Variables**
   ```env
   NODE_ENV=development
   PORT=5000
   MONGODB_URI=mongodb://localhost:27017/recursia-dx
   JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
   ```

4. **Start Development Server**
   ```bash
   npm run dev
   ```

The server will start on `http://localhost:5000`

## API Documentation

### Authentication Endpoints
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current user
- `PUT /api/auth/profile` - Update user profile
- `PUT /api/auth/change-password` - Change password

### Sample Management
- `POST /api/samples` - Create new sample
- `GET /api/samples` - List samples with filtering
- `GET /api/samples/:id` - Get sample details
- `PUT /api/samples/:id` - Update sample
- `PUT /api/samples/:id/status` - Update sample status
- `POST /api/samples/:id/images` - Upload sample images

### Analysis Management
- `POST /api/analysis` - Create new analysis
- `GET /api/analysis` - List analyses
- `GET /api/analysis/:id` - Get analysis details
- `PUT /api/analysis/:id` - Update analysis
- `POST /api/analysis/:id/validate` - Validate analysis results

### Report Generation
- `POST /api/reports` - Create new report
- `GET /api/reports` - List reports
- `GET /api/reports/:id` - Get report details
- `PUT /api/reports/:id` - Update report
- `POST /api/reports/:id/approve` - Approve report
- `POST /api/reports/:id/review` - Add report review

## Database Models

### User Model
- Authentication and profile information
- Role-based permissions
- Login attempt tracking and account lockout
- Preferences and settings

### Sample Model
- Patient information (anonymized)
- Specimen details and metadata
- Workflow status tracking
- Image attachments and processing history
- Quality control checkpoints

### Analysis Model
- AI model results and confidence scores
- Multiple analysis types support
- Validation workflow and reviewer assignments
- Performance metrics and accuracy tracking

### Report Model
- Structured pathology reports
- Template-based content organization
- Digital signatures and approval workflow
- Version control and amendment tracking
- Distribution and delivery management

## Security Features

### Authentication Security
- Bcrypt password hashing with configurable salt rounds
- JWT tokens with secure signing and expiration
- Account lockout after failed login attempts
- Rate limiting on authentication endpoints

### Data Protection
- Input validation and sanitization
- SQL injection prevention through Mongoose ODM
- XSS protection with security headers
- CORS configuration for cross-origin requests
- Audit trails for sensitive operations

### Access Control
- Role-based permissions system
- Resource-level authorization checks
- User-specific data filtering
- Admin-level access controls

## Development

### Scripts
```bash
npm start          # Start production server
npm run dev        # Start development server with nodemon
npm test           # Run test suite
npm run lint       # Run ESLint
npm run format     # Format code with Prettier
```

### Environment Configurations
- **Development**: Debug logging, relaxed security
- **Production**: Optimized performance, strict security
- **Test**: Isolated database, minimal logging

## Deployment

### Production Checklist
- [ ] Set strong JWT_SECRET in environment
- [ ] Configure production MongoDB URI
- [ ] Set up SSL/TLS certificates
- [ ] Configure proper CORS origins
- [ ] Set up monitoring and logging
- [ ] Configure file upload storage
- [ ] Set up backup procedures

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 5000
CMD ["npm", "start"]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API examples

---

Built with ❤️ for the digital pathology community