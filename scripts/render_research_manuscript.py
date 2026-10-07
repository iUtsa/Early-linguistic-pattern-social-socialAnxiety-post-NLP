"""Render the human-review manuscript with existing Pandoc/LaTeX tools."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    if not shutil.which('pandoc') or not shutil.which('pdflatex'):
        raise SystemExit('Install Pandoc and a pdflatex distribution to render this draft.')
    research = root / 'research'
    source = research / 'REVISED_PILOT_MANUSCRIPT.md'
    outputs = [research / f'REVISED_PILOT_MANUSCRIPT.{suffix}' for suffix in ['pdf', 'docx']]
    # Inline figures avoid float/longtable overlap in this working draft.
    common = ['pandoc', str(source), '--from=markdown-implicit_figures', '--standalone',
              f'--resource-path={research}']
    subprocess.run(common + ['--pdf-engine=pdflatex', '-V', 'geometry:margin=0.75in',
                             '-V', 'papersize:a4', '-V', 'fontsize:10pt',
                             '-V', 'colorlinks:true', '-V', 'linkcolor:blue',
                             '-o', str(outputs[0])], cwd=root, check=True)
    subprocess.run(common + ['-o', str(outputs[1])], cwd=root, check=True)
    inputs = [source, research / 'figures/reviewer_controls.png',
              research / 'figures/temporal_extension.png', Path(__file__)]
    manifest = {
        'status': 'Human-review draft; ethics details and author declarations incomplete',
        'pandoc_version': subprocess.check_output(['pandoc', '--version'], text=True).splitlines()[0],
        'pdflatex_version': subprocess.check_output(['pdflatex', '--version'], text=True).splitlines()[0],
        'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in inputs + outputs},
    }
    (research / 'manuscript_export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Rendered PDF and editable DOCX draft; source/output hashes recorded.')


if __name__ == '__main__':
    main()
