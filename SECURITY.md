# Security Policy

escha-mlx is an inference runtime: it loads model checkpoints and serves an HTTP
endpoint. The realistic security surface is (a) parsing untrusted checkpoint files and
(b) the OpenAI-compatible server, which is intended for localhost/LAN use and ships with
no authentication — do not expose it directly to the public internet.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/EschaLabs/escha-mlx/security/advisories/new)
rather than a public issue.
We will acknowledge within a few days and coordinate a fix and disclosure with you.

## Supported versions

Only the latest release receives security fixes.
