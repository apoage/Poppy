# audit_documentation.py

**Documentation Audit Tool**

## Purpose

Audit documentation coverage across the codebase.

## Usage

```bash
python tools/audit_documentation.py
```

## Checks

| Category | Checks |
|----------|--------|
| Docstrings | Functions with/without docstrings |
| README | README.md presence |
| Comments | Comment ratio |
| Type hints | Type annotation coverage |

## Example Output

```
Documentation Audit
===================
Docstrings:
  Functions: 156 total
  With docstrings: 89 (57%)
  Missing: 67 (43%)

Files:
  README.md: ✅ Present
  CLAUDE.md: ✅ Present

Comments:
  Ratio: 12% (good)

Type hints:
  Coverage: 45%
```

## Dependencies

- ast (Python AST parsing)
- os

## Related Files

- `audit_codebase.py` - Code quality audit
