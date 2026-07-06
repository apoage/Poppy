# audit_codebase.py

**Codebase Audit Tool**

## Purpose

Audit the codebase for quality metrics and potential issues.

## Usage

```bash
python tools/audit_codebase.py
```

## Checks

| Category | Checks |
|----------|--------|
| Structure | File count, line count |
| Quality | TODO count, FIXME count |
| Imports | Unused imports |
| Security | Hardcoded secrets |
| Style | Long lines, complexity |

## Example Output

```
Codebase Audit
==============
Files: 45 Python files
Lines: 12,340 total

Quality:
  TODOs: 23
  FIXMEs: 5
  Long lines (>120): 12

Security:
  ✅ No hardcoded secrets found

Complexity:
  High (>10): 3 functions
  Medium (5-10): 15 functions
```

## Dependencies

- os, re

## Related Files

- `audit_documentation.py` - Doc audit
