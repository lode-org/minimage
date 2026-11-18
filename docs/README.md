# minimage docs

Org under `orgmode/` is the source. `emacs --batch -l docs/export.el`
writes HTML into `website/`. GitHub Pages serves that tree.

Do not edit the HTML by hand.

The export is deterministic. Anchor ids come from the heading path, not
from Emacs's random number generator, so regenerating unchanged Org
leaves the HTML byte for byte the same. `scripts/check-docs.py` rebuilds
the pages and fails on any byte difference with the tracked HTML.
