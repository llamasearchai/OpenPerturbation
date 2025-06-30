# Security Policy

## Supported Versions

We actively support the following versions of OpenPerturbation:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of OpenPerturbation seriously. If you discover a security vulnerability, please follow these steps:

### Private Disclosure Process

1. **Do not** create a public GitHub issue for security vulnerabilities
2. Send an email to **nikjois@llamasearch.ai** with:
   - A clear description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes (if available)

### Response Timeline

- **Initial Response**: Within 48 hours of report
- **Status Update**: Within 7 days with preliminary assessment
- **Resolution**: Target 30 days for critical issues, 90 days for others

### Disclosure Policy

We follow coordinated disclosure:
- We will work with you to understand and resolve the issue
- We will acknowledge your contribution (with your permission)
- We will not pursue legal action against good-faith security researchers

### Security Best Practices

When using OpenPerturbation in production:
- Keep dependencies updated
- Use HTTPS for all API communications
- Implement proper authentication and authorization
- Monitor logs for suspicious activity
- Follow the principle of least privilege

### Contact

For security-related questions: **nikjois@llamasearch.ai**

---

**Author**: Nik Jois  
**Email**: nikjois@llamasearch.ai

## Security Features

OpenPerturbation includes several built-in security features:

### API Security
- **Input Validation**: All API inputs are validated using Pydantic models
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **CORS Configuration**: Configurable CORS policies
- **Request Size Limits**: Maximum request size limits to prevent DoS attacks
- **Error Handling**: Secure error messages that don't leak sensitive information

### Data Security
- **File Upload Validation**: Strict validation of uploaded files
- **Path Traversal Protection**: Prevention of directory traversal attacks
- **Temporary File Cleanup**: Automatic cleanup of temporary files
- **Data Sanitization**: Input sanitization for all user-provided data

### Infrastructure Security
- **Docker Security**: Multi-stage Docker builds with minimal attack surface
- **Non-root User**: Container runs as non-root user
- **Dependency Scanning**: Automated vulnerability scanning of dependencies
- **Security Headers**: Appropriate security headers in HTTP responses

## Vulnerability Disclosure Timeline

We follow a coordinated disclosure timeline:

1. **Day 0**: Vulnerability reported
2. **Day 1-3**: Initial assessment and acknowledgment
3. **Day 3-7**: Detailed analysis and fix development
4. **Day 7-14**: Testing and validation of fix
5. **Day 14-21**: Release preparation and coordination
6. **Day 21**: Public disclosure and release

## Security Contact

For security-related questions or concerns:

- **Email**: nikjois@llamasearch.ai
- **Subject**: [SECURITY] OpenPerturbation Security Issue
- **PGP Key**: Available upon request

## Bug Bounty Program

While we don't currently offer monetary rewards, we do offer:

- Public acknowledgment of your contribution (if desired)
- Direct communication with the development team
- Early access to new features and releases
- Contribution to the security of the scientific computing community

## Legal

This security policy is provided under the same MIT license as the OpenPerturbation project. By reporting vulnerabilities, you agree to:

- Give us reasonable time to investigate and mitigate the issue before public disclosure
- Make a good faith effort to avoid privacy violations, destruction of data, and interruption or degradation of services
- Only interact with accounts you own or with explicit permission of the account holder

## Acknowledgments

We would like to thank the following security researchers for their responsible disclosure:

*(No reports received yet - be the first!)*

---

**Last Updated**: January 20, 2024  
**Version**: 1.0

For general questions about OpenPerturbation, please use the [GitHub Issues](https://github.com/llamasearchai/OpenPerturbation/issues) page. 