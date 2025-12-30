# Security Policy

## 🔒 Security Overview

KnoAD-Net is a research project for Alzheimer's Disease detection. While this is not production medical software, we take security seriously, especially regarding:

- **Patient Data Privacy**: HIPAA/GDPR compliance when handling medical data
- **Model Security**: Preventing adversarial attacks or model manipulation
- **Data Integrity**: Ensuring research reproducibility and preventing data corruption
- **Dependency Security**: Keeping libraries up-to-date and vulnerability-free

---

## 🎯 Scope

### In Scope

The following are within the scope of our security policy:

- **Code Security**: Vulnerabilities in Python code, dependencies, or scripts
- **Data Privacy**: Issues that could lead to patient data exposure
- **Model Security**: Vulnerabilities in model loading, inference, or training
- **Authentication/Authorization**: If deployment features are added
- **Dependency Vulnerabilities**: Known CVEs in required packages
- **Input Validation**: Improper handling of user inputs or file uploads

### Out of Scope

The following are explicitly **NOT** covered by this security policy:

- **Clinical Validation**: This is research software, not a medical device
- **Diagnostic Accuracy**: Lower-than-expected accuracy is not a security issue
- **Social Engineering**: Attacks targeting users directly (not the software)
- **Physical Security**: Hardware theft, physical access to systems
- **Denial of Service**: Resource exhaustion attacks (research context only)

---

## 🚨 Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 1.0.x   | ✅ Yes             | Current stable release |
| < 1.0   | ❌ No              | Development versions (upgrade required) |

---

## 🛡️ Reporting a Vulnerability

### How to Report

If you discover a security vulnerability, please follow these steps:

#### 1. **DO NOT** Create a Public Issue

For security vulnerabilities, **DO NOT** open a public GitHub issue. This could put users at risk.

#### 2. Report Privately

**Preferred Method**: Use GitHub's private vulnerability reporting feature

1. Go to the [Security tab](https://github.com/Aspect022/Alzimers-Research/security)
2. Click "Report a vulnerability"
3. Fill out the form with details

**Alternative Method**: Email the maintainers directly

- **Email**: [Create an issue with email request - maintainers will provide email]
- **Subject**: "SECURITY: [Brief description]"

#### 3. Include Required Information

Your report should include:

- **Description**: Clear description of the vulnerability
- **Impact**: What could an attacker do with this vulnerability?
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Affected Components**: Which files, functions, or modules are affected?
- **Potential Fix**: Suggestions for remediation (if you have them)
- **CVE/CWE References**: If applicable

**Example Report:**

```markdown
**Summary**: SQL Injection in custom data loader (hypothetical)

**Impact**: Attacker could execute arbitrary SQL if using custom database backend

**Steps to Reproduce**:
1. Modify config.py to use database backend
2. Craft malicious input: `subject_id = "1' OR '1'='1"`
3. Run data pipeline
4. Observe unauthorized data access

**Affected Component**: phase1_data_pipeline.py, line 123

**Suggested Fix**: Use parameterized queries instead of string formatting

**Severity**: High (CVSS 7.5)
```

---

## ⏱️ Response Timeline

### What to Expect

1. **Acknowledgment**: Within 48 hours of report
2. **Initial Assessment**: Within 5 business days
3. **Status Update**: Every 7 days until resolved
4. **Resolution**: Depends on severity (see below)

### Severity Levels

| Severity | Response Time | Fix Timeline | Examples |
|----------|--------------|--------------|----------|
| **Critical** | 24 hours | 7 days | Remote code execution, patient data exposure |
| **High** | 48 hours | 14 days | Authentication bypass, sensitive data leakage |
| **Medium** | 5 days | 30 days | Input validation issues, dependency vulnerabilities |
| **Low** | 10 days | 60 days | Information disclosure (non-sensitive) |

---

## 🔐 Security Best Practices

### For Users

If you're using KnoAD-Net, follow these security best practices:

#### Data Privacy

- **Never commit patient data** to Git repositories
- **Anonymize datasets** before processing
- **Use encryption** for data at rest and in transit
- **Comply with regulations**: HIPAA (US), GDPR (EU), local laws
- **Obtain consent**: Ensure proper patient consent for research use

#### Environment Security

- **Use virtual environments**: Isolate dependencies
- **Keep dependencies updated**: Run `pip install --upgrade -r requirements.txt`
- **Review new dependencies**: Check for known vulnerabilities before adding
- **Limit access**: Restrict who can access systems with medical data

#### Model Security

- **Verify checksums**: Ensure model weights haven't been tampered with
- **Secure model storage**: Protect trained models from unauthorized access
- **Validate inputs**: Sanitize MRI files and clinical data before processing
- **Monitor predictions**: Watch for unusual model behavior

### For Contributors

If you're contributing to KnoAD-Net:

#### Code Security

- **Input validation**: Always validate and sanitize user inputs
- **Avoid hardcoded secrets**: Use environment variables for sensitive data
- **Use safe deserialization**: Be careful with `pickle`, `torch.load`, etc.
- **Dependency scanning**: Check new dependencies for vulnerabilities
- **Code review**: Have security-minded reviewers check your code

```python
# ❌ Unsafe
model = torch.load(user_provided_path)  # Could load malicious model

# ✅ Safe
if os.path.exists(trusted_model_path):
    model = torch.load(trusted_model_path, map_location='cpu')
else:
    raise ValueError("Model file not found at trusted location")
```

#### Data Handling

```python
# ❌ Unsafe - SQL Injection risk
query = f"SELECT * FROM patients WHERE id = '{user_input}'"

# ✅ Safe - Parameterized query
query = "SELECT * FROM patients WHERE id = %s"
cursor.execute(query, (user_input,))
```

```python
# ❌ Unsafe - Path traversal risk
file_path = f"/data/{user_filename}"
with open(file_path, 'r') as f:
    data = f.read()

# ✅ Safe - Validate and sanitize
safe_filename = os.path.basename(user_filename)  # Remove path components
file_path = os.path.join(TRUSTED_DIR, safe_filename)
if not file_path.startswith(TRUSTED_DIR):
    raise ValueError("Invalid file path")
with open(file_path, 'r') as f:
    data = f.read()
```

---

## 🔍 Known Vulnerabilities

### Current Status

✅ **No known security vulnerabilities** as of December 2025.

### Dependency Vulnerabilities

We actively monitor dependencies for known CVEs. Check our latest scan:

```bash
# Run security audit
pip install safety
safety check -r requirements.txt
```

### Past Vulnerabilities

None at this time. When addressed, they will be listed here with:
- CVE number (if assigned)
- Affected versions
- Fixed version
- Mitigation steps

---

## 📋 Security Checklist for Medical AI

Given the sensitive nature of medical AI, we follow these additional guidelines:

### Data Privacy ✅

- [ ] No patient identifiable information (PII) in code or docs
- [ ] Data anonymization guidelines provided
- [ ] HIPAA/GDPR compliance considerations documented
- [ ] Secure data storage recommendations provided

### Model Integrity ✅

- [ ] Model checksum verification
- [ ] Secure model loading practices
- [ ] Input validation for inference
- [ ] Adversarial robustness considerations

### Research Ethics ✅

- [ ] Clear "research use only" disclaimers
- [ ] Proper dataset attribution (OASIS-1)
- [ ] Informed consent requirements documented
- [ ] IRB approval guidance provided

### Transparency ✅

- [ ] Model limitations clearly documented
- [ ] Bias and fairness considerations addressed
- [ ] Explainability features (RAG module)
- [ ] Performance metrics transparently reported

---

## 🛠️ Security Tools

### Recommended Tools

We recommend using these tools for security scanning:

```bash
# Dependency vulnerability scanning
pip install safety
safety check

# Code security analysis
pip install bandit
bandit -r . -ll

# Secret detection
pip install detect-secrets
detect-secrets scan
```

### Continuous Security

For contributors, we encourage:

- Pre-commit hooks for security checks
- Regular dependency updates
- Code review with security focus
- Threat modeling for new features

---

## 📞 Contact

### Security Team

- **GitHub**: Use private vulnerability reporting
- **Response Time**: See [Response Timeline](#response-timeline)

### General Security Questions

For non-sensitive security questions:
- Open a GitHub Discussion
- Tag with "security" label

---

## 🎓 Responsible Disclosure

We follow responsible disclosure practices:

1. **Private Reporting**: Vulnerabilities reported privately first
2. **Coordinated Disclosure**: Fix developed and tested privately
3. **Public Disclosure**: Announced after fix is released
4. **Credit**: Reporters acknowledged (if desired)

---

## 📜 Legal

### Disclaimer

This software is provided "as is" for research purposes only. See [LICENSE](LICENSE) for full terms.

**NO WARRANTY**: We make no guarantees about security or fitness for clinical use.

**NO LIABILITY**: We are not liable for any damages resulting from security vulnerabilities.

---

## 🏆 Hall of Fame

We recognize security researchers who help improve KnoAD-Net:

<!-- Contributors who report valid security issues will be listed here -->

*No security reports yet. Be the first!*

---

## 📚 Additional Resources

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [FDA Guidance on AI/ML Medical Devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)

---

**Last Updated**: December 2025  
**Version**: 1.0  
**Next Review**: June 2026
