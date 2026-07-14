"""Generate sitemap.xml into _build/html after `jupyter-book build .`."""
import os

BASE = "https://mikenguyen13.github.io/mlpy/"
ROOT = "_build/html"
SKIP = {"genindex.html", "search.html", "404.html"}

urls = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel = os.path.relpath(dirpath, ROOT)
    parts = [] if rel == "." else rel.split(os.sep)
    if any(p.startswith(("_", ".")) for p in parts):
        continue
    for f in filenames:
        if f.endswith(".html") and f not in SKIP:
            urls.append(BASE + "/".join(parts + [f]))

urls.sort()
with open(os.path.join(ROOT, "sitemap.xml"), "w", encoding="utf-8") as fh:
    fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    fh.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
    for u in urls:
        fh.write(f"  <url><loc>{u}</loc></url>\n")
    fh.write("</urlset>\n")
print(f"sitemap.xml written with {len(urls)} urls")
