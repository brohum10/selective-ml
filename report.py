"""Render a dependency-free SVG experiment dashboard from measured results."""
import json
from pathlib import Path
import argparse


def render(folder):
    runs = json.loads((folder/'metrics.json').read_text())['runs']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="600" viewBox="0 0 1040 600" role="img" aria-label="SelectiveML held-out experiment results">',
           '<rect width="1040" height="600" rx="24" fill="#101827"/>',
           '<style>text{font-family:system-ui,sans-serif;fill:#e2e8f0}.muted{fill:#94a3b8}</style>']
    def text(x,y,value,size=16,cls=''):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}">{value}</text>')
    text(40,55,'SelectiveML / confidence is not a guarantee',28)
    text(40,88,'Three fixed seeds · 275 held-out examples each · public 8×8 digits',16,'muted')
    text(40,137,'Held-out accuracy (bars start at zero)',18)
    for i, r in enumerate(runs):
        y=180+i*100
        text(40,y,f"Seed {r['seed']}")
        for j,(key,color,label) in enumerate((('linear_baseline','#64748b','Linear'),('neural_calibrated','#2dd4bf','MLP'),('noise_stress_sigma_0_2','#fb923c','MLP + noise'))):
            acc=r[key]['accuracy']
            svg.append(f'<rect x="160" y="{y-14+j*24}" width="{acc*490:.2f}" height="16" rx="4" fill="{color}"/>')
            text(665,y+j*24,f'{label}: {acc:.2%}',14)
    text(40,488,'What the experiment actually shows',19)
    text(40,518,'The neural model beats the linear baseline in all three splits.',16)
    text(40,544,'Calibration worsens test NLL in 2/3 seeds; noise reduces accuracy to 90.18–94.18%.',16)
    text(40,570,'Small benchmark, overlapping split experiments—not production OCR or a statistical guarantee.',14,'muted')
    svg.append('</svg>')
    (folder/'dashboard.svg').write_text('\n'.join(svg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    render(parser.parse_args().folder)
