# Security Policy

## Supported Versions

We actively support the following versions of OpenPerturbation with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The OpenPerturbation team takes security seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**nikjois@llamasearch.ai**

Include the following information in your report:
- Type of issue (e.g., XSS, SQL injection, remote code execution, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Initial Assessment**: We will provide an initial assessment within 72 hours.
- **Regular Updates**: We will keep you informed about our progress throughout the process.
- **Resolution Timeline**: We aim to resolve critical vulnerabilities within 7 days, high-severity issues within 30 days, and medium/low severity issues within 90 days.

### Our Commitment

- We will respond to your report promptly and work with you to understand and resolve the issue quickly.
- We will keep you informed about our progress throughout the process.
- We will credit you for your discovery (unless you prefer to remain anonymous).
- We will not pursue legal action against researchers who discover and report vulnerabilities responsibly.

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest version of OpenPerturbation
2. **Environment Variables**: Store sensitive configuration in environment variables, not in code
3. **Network Security**: Run the API behind a reverse proxy (nginx, Apache) in production
4. **Access Control**: Implement proper authentication and authorization for your deployment
5. **HTTPS**: Always use HTTPS in production environments
6. **Input Validation**: Validate all inputs when using the API programmatically

### For Developers

1. **Dependencies**: Regularly update dependencies and monitor for known vulnerabilities
2. **Code Review**: All code changes require review before merging
3. **Static Analysis**: Use security linters (bandit) as part of the CI/CD pipeline
4. **Testing**: Include security testing in the test suite
5. **Secrets**: Never commit secrets, API keys, or credentials to version control

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