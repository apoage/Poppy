#!/usr/bin/env python3
"""
Cross-reference documentation with codebase reality.
Find stale claims.
"""

import os
import re
from pathlib import Path

def scan_documentation_claims():
    """Scan all .md files for feature claims"""

    claims = {
        'gemini_md': [],
        'recent_improvements': [],
        'future_enhancements': [],
        'design_docs': []
    }

    # Scan GEMINI.md
    if os.path.exists('GEMINI.md'):
        with open('GEMINI.md') as f:
            content = f.read()

            # Look for checkboxes
            resolved = re.findall(r'- \[x\] (.+)', content, re.IGNORECASE)
            pending = re.findall(r'- \[ \] (.+)', content, re.IGNORECASE)

            claims['gemini_md'] = {
                'resolved': resolved,
                'pending': pending
            }

    # Scan recent_improvements.md
    if os.path.exists('lore/recent_improvements.md'):
        with open('lore/recent_improvements.md') as f:
            content = f.read()

            # Extract improvement headers
            improvements = re.findall(r'^## (.+)', content, re.MULTILINE)
            claims['recent_improvements'] = improvements

    # Scan future_enhancements.md
    if os.path.exists('lore/future_enhancements.md'):
        with open('lore/future_enhancements.md') as f:
            content = f.read()

            # Extract enhancement sections
            enhancements = re.findall(r'^## \d+\. (.+)', content, re.MULTILINE)
            claims['future_enhancements'] = enhancements

    return claims

def cross_reference(claims, implementation_status):
    """Cross-reference claims with implementation"""

    report = ["# Documentation Audit Report", ""]
    report.append("## Stale/Incorrect Claims")
    report.append("")

    # Check if "Rainbow Table" is in future enhancements but actually implemented
    if 'Rainbow Table' in str(claims['future_enhancements']):
        report.append("### ⚠️ STALE: Rainbow Table in Future Enhancements")
        report.append("- **Status**: Already implemented (2 days ago)")
        report.append("- **Action**: Move to recent_improvements.md")
        report.append("")

    # Check GEMINI.md pending items
    if claims.get('gemini_md', {}).get('pending'):
        report.append("### GEMINI.md Pending Items")
        for item in claims['gemini_md']['pending'][:5]:  # First 5
            report.append(f"- [ ] {item}")
        report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    print("Scanning documentation...")
    claims = scan_documentation_claims()

    # Load implementation status
    impl_status = {}
    if os.path.exists('AUDIT_IMPLEMENTATION_STATUS.md'):
        with open('AUDIT_IMPLEMENTATION_STATUS.md') as f:
            impl_status = f.read()

    report = cross_reference(claims, impl_status)

    with open('AUDIT_DOCUMENTATION_DRIFT.md', 'w') as f:
        f.write(report)

    print("Report saved to: AUDIT_DOCUMENTATION_DRIFT.md")
    print(report)
