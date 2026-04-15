"""Generate a printable A4 PDF with 3.1cm top-face squares for sim-to-real cube testing."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

# A4 in inches
A4_W, A4_H = 8.27, 11.69

# Cube side length in cm -> inches
SIDE_CM = 3.1
SIDE_IN = SIDE_CM / 2.54

squares = [
    ('#00BFFF', 'Bright Blue'),   # deep sky blue
    ('#00FF7F', 'Bright Green'),  # spring green
]

with PdfPages('cube_template.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(A4_W, A4_H))
    ax.set_xlim(0, A4_W)
    ax.set_ylim(0, A4_H)
    ax.set_aspect('equal')
    ax.axis('off')

    spacing = 0.5
    total_w = len(squares) * SIDE_IN + (len(squares) - 1) * spacing
    start_x = (A4_W - total_w) / 2
    start_y = (A4_H - SIDE_IN) / 2

    for i, (color, label) in enumerate(squares):
        x = start_x + i * (SIDE_IN + spacing)
        rect = patches.Rectangle(
            (x, start_y), SIDE_IN, SIDE_IN,
            linewidth=1.5, edgecolor='black', facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + SIDE_IN / 2, start_y - 0.15, f'{label}\n{SIDE_CM}cm × {SIDE_CM}cm',
                ha='center', va='top', fontsize=9, color='#222222')

    ax.text(A4_W / 2, A4_H - 0.25, f'Sim-to-Real Cube Top Face — {SIDE_CM}cm — Print at 100% scale',
            ha='center', va='top', fontsize=10, fontweight='bold', color='#222222')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print("Saved cube_template.pdf")
