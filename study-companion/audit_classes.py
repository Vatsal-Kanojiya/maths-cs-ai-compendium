"""Every class used in a sheet's markup must be defined in that sheet's own CSS.

Chapters 09-13 were built on header and navigation classes the design system never defines
(sheet, tb-main, tb-meta, tb-sub, toc, lede), copied forward from page to page. No layout
probe noticed, because nothing looked. Run this before every publish:

    python3 study-companion/audit_classes.py

A clean sheet prints its name and nothing else. Exit status is 1 if any sheet has an
undefined class.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
bad = 0
for path in sorted(glob.glob(os.path.join(HERE, 'ch*', 'index.html'))):
    src = open(path, encoding='utf-8').read()
    css = ' '.join(re.findall(r'<style>(.*?)</style>', src, re.S))
    markup = re.sub(r'<script>.*?</script>', '', src, flags=re.S)
    used = set()
    for attr in re.findall(r'class="([^"]+)"', markup):
        used.update(attr.split())
    undefined = sorted(c for c in used
                       if not re.search(r'\.' + re.escape(c) + r'(?![\w-])', css))
    name = os.path.basename(os.path.dirname(path))
    print(f'{name:34s} {" ".join(undefined)}')
    bad += bool(undefined)
sys.exit(1 if bad else 0)
