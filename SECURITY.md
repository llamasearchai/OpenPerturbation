# Security Policy

## Supported Versions

We actively support the following versions of OpenPerturbation with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | ✅ Yes             |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## API Key and Secrets Management

### Environment Variables

OpenPerturbation uses environment variables for sensitive configuration. **Never hardcode API keys or secrets in your code.**

#### Secure Setup Process

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual values:**
   ```bash
   # Required for AI-powered features
   OPENAI_API_KEY=sk-your-actual-api-key-here
   
   # Optional: Database connections
   DATABASE_URL=postgresql://user:password@localhost:5432/db
   ```

3. **Verify `.env` is in `.gitignore`:**
   ```bash
   grep -q "^\.env$" .gitignore && echo "✅ .env is ignored" || echo "❌ Add .env to .gitignore"
   ```

#### Production Deployment

For production environments:

- **Use environment variables directly** (not `.env` files)
- **Use secrets management services** (AWS Secrets Manager, Azure Key Vault, etc.)
- **Rotate API keys regularly** (every 90 days minimum)
- **Use least-privilege access** for service accounts

#### Key Resolution Priority

The system resolves API keys in this order:

1. **Explicit constructor argument** (for testing only)
2. **Environment variable** (`OPENAI_API_KEY`)
3. **`.env` file** (development only)
4. **Fallback to mock mode** (safe degradation)

### Container Security

When using Docker:

```dockerfile
# Use build-time args for non-secret config only
ARG APP_VERSION=1.1.0

# Use runtime environment for secrets
ENV OPENAI_API_KEY=""
```

```bash
# Pass secrets at runtime
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" openperturbation
```

### CI/CD Security

For GitHub Actions and other CI systems:

- Store secrets in repository/organization secrets
- Use encrypted secrets, never plain text
- Limit secret access to specific branches/environments
- Audit secret usage regularly

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please follow responsible disclosure:

### How to Report

1. **Email:** Send details to [nikjois@llamasearch.ai](mailto:nikjois@llamasearch.ai)
2. **Subject:** Use "SECURITY: [Brief Description]"
3. **Include:**
   - Vulnerability description
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if known)

### What to Expect

- **Initial Response:** Within 48 hours
- **Investigation:** 1-7 days depending on complexity
- **Fix Timeline:** Critical issues within 7 days, others within 30 days
- **Disclosure:** Coordinated disclosure after fix is available

### Scope

Security issues we consider in scope:

- ✅ API key exposure or leakage
- ✅ Authentication/authorization bypasses
- ✅ Code injection vulnerabilities
- ✅ Privilege escalation
- ✅ Data exposure or corruption
- ✅ Denial of service attacks

Out of scope:

- ❌ Social engineering attacks
- ❌ Physical security issues
- ❌ Issues in third-party dependencies (report to them directly)
- ❌ Self-inflicted issues from misconfigurations

## Security Best Practices

### For Developers

1. **Never commit secrets** to version control
2. **Use type hints** for security-sensitive functions
3. **Validate all inputs** from external sources
4. **Use parameterized queries** for database operations
5. **Enable logging** for security events
6. **Keep dependencies updated** regularly

### For Users

1. **Use strong, unique API keys** for each environment
2. **Enable two-factor authentication** on all accounts
3. **Regularly audit** API key usage and permissions
4. **Monitor logs** for suspicious activity
5. **Use HTTPS** for all API communications
6. **Implement rate limiting** in production

### Code Security

```python
# ✅ Good: Secure API key handling
from openperturbation.agents import OpenPerturbationAgent

# Key resolved from environment/config
agent = OpenPerturbationAgent()

# ❌ Bad: Hardcoded secrets
agent = OpenPerturbationAgent(api_key="sk-hardcoded-key")
```

## Security Monitoring

We recommend implementing:

- **API rate limiting** (built into FastAPI endpoints)
- **Request logging** with sanitized sensitive data
- **Error monitoring** (Sentry integration available)
- **Dependency scanning** (automated via GitHub Actions)

## Compliance

OpenPerturbation is designed to support:

- **GDPR** compliance for EU data processing
- **HIPAA** guidelines for healthcare data (with proper configuration)
- **SOC 2** controls for service organizations
- **ISO 27001** information security standards

## Security Updates

- Subscribe to GitHub releases for security announcements
- Enable Dependabot alerts for dependency vulnerabilities
- Follow [@OpenPerturbation](https://twitter.com/openperturbation) for security advisories

---

**Last Updated:** January 18, 2025  
**Version:** 1.1.0

For general questions about OpenPerturbation, please use the [GitHub Issues](https://github.com/llamasearchai/OpenPerturbation/issues) page. 