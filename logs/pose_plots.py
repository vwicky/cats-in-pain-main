"""Generate cat pose keypoint figure for scientific paper."""

COLORS = {
    "head":  "#E07040",
    "face":  "#D04040",
    "spine": "#3E80C0",
    "neck":  "#208E78",
    "front": "#3EA060",
    "back":  "#9050B0",
    "body":  "#909090",
    "na":    "#CCCCCC",
}
CAT_FILL   = "#E2DED8"
CAT_STROKE = "#888"
CAT_SW     = "0.8"

# -------- ViTPose 17 keypoints (x, y, label, group) --------
VIT = [
    (74, 55,  "L_Eye",        "head"),
    (88, 51,  "R_Eye",        "head"),
    (108,63,  "Nose",         "head"),
    (90, 90,  "Neck",         "spine"),
    (258,108, "Root of tail", "spine"),
    (115,125, "L_Shoulder",   "front"),
    (118,162, "L_Elbow",      "front"),
    (120,205, "L_F_Paw",      "front"),
    (99, 122, "R_Shoulder",   "front"),
    (97, 159, "R_Elbow",      "front"),
    (95, 202, "R_F_Paw",      "front"),
    (218,127, "L_Hip",        "back"),
    (222,164, "L_Knee",       "back"),
    (224,206, "L_B_Paw",      "back"),
    (201,124, "R_Hip",        "back"),
    (203,161, "R_Knee",       "back"),
    (202,203, "R_B_Paw",      "back"),
]
VIT_SKEL = [
    (0,2,"head"),(1,2,"head"),(0,3,"head"),(1,3,"head"),
    (3,4,"spine"),
    (3,8,"front"),(8,9,"front"),(9,10,"front"),
    (3,5,"front"),(5,6,"front"),(6,7,"front"),
    (4,14,"back"),(14,15,"back"),(15,16,"back"),
    (4,11,"back"),(11,12,"back"),(12,13,"back"),
]

# -------- SuperAnimal 39 keypoints --------
SA = [
    (108,63,  "nose",               "face"),
    (103,68,  "upper_jaw",          "face"),
    (103,73,  "lower_jaw",          "face"),
    (98, 68,  "mouth_end_right",    "face"),
    (96, 71,  "mouth_end_left",     "face"),
    (88, 51,  "right_eye",          "head"),
    (72, 43,  "right_earbase",      "head"),
    (63, 17,  "right_earend",       "head"),
    (None,None,"right_antler_base", "na"),
    (None,None,"right_antler_end",  "na"),
    (74, 55,  "left_eye",           "head"),
    (66, 45,  "left_earbase",       "head"),
    (56, 21,  "left_earend",        "head"),
    (None,None,"left_antler_base",  "na"),
    (None,None,"left_antler_end",   "na"),
    (60, 82,  "neck_base",          "neck"),
    (90, 90,  "neck_end",           "neck"),
    (65, 88,  "throat_base",        "neck"),
    (81, 104, "throat_end",         "neck"),
    (97, 88,  "back_base",          "spine"),
    (253,84,  "back_end",           "spine"),
    (175,78,  "back_middle",        "spine"),
    (258,108, "tail_base",          "spine"),
    (246,40,  "tail_end",           "spine"),
    (115,125, "front_left_thai",    "front"),
    (118,162, "front_left_knee",    "front"),
    (120,205, "front_left_paw",     "front"),
    (99, 122, "front_right_thai",   "front"),
    (97, 159, "front_right_knee",   "front"),
    (95, 202, "front_right_paw",    "front"),
    (224,206, "back_left_paw",      "back"),
    (218,127, "back_left_thai",     "back"),
    (201,124, "back_right_thai",    "back"),
    (222,164, "back_left_knee",     "back"),
    (203,161, "back_right_knee",    "back"),
    (202,203, "back_right_paw",     "back"),
    (175,168, "belly_bottom",       "body"),
    (178,104, "body_middle_right",  "body"),
    (170,108, "body_middle_left",   "body"),
]
SA_SKEL = [
    (0,1,"face"),(1,3,"face"),(0,2,"face"),(2,4,"face"),
    (0,5,"head"),(0,10,"head"),
    (5,6,"head"),(6,7,"head"),
    (10,11,"head"),(11,12,"head"),
    (5,15,"neck"),(10,15,"neck"),
    (15,16,"neck"),(15,17,"neck"),(17,18,"neck"),(16,18,"neck"),
    (16,19,"spine"),(18,19,"spine"),
    (19,21,"spine"),(21,20,"spine"),(20,22,"spine"),(22,23,"spine"),
    (19,27,"front"),(27,28,"front"),(28,29,"front"),
    (19,24,"front"),(24,25,"front"),(25,26,"front"),
    (22,32,"back"),(32,34,"back"),(34,35,"back"),
    (22,31,"back"),(31,33,"back"),(33,30,"back"),
    (36,37,"body"),(36,38,"body"),(37,38,"body"),
    (19,37,"body"),(20,38,"body"),
]


def cat_svg(ox, oy):
    """Cat silhouette shapes (all in local 0-300, 0-220 space, offset by ox,oy)."""
    def p(x,y): return f"{ox+x},{oy+y}"
    def f(x,y): return (ox+x, oy+y)
    S = []
    cf, cs = CAT_FILL, CAT_STROKE

    # Body
    S.append(f'<ellipse cx="{ox+178}" cy="{oy+130}" rx="92" ry="43" fill="{cf}" stroke="{cs}" stroke-width="{CAT_SW}"/>')
    # Far ear
    S.append(f'<polygon points="{p(87,42)} {p(97,17)} {p(110,42)}" fill="{cf}" stroke="{cs}" stroke-width="{CAT_SW}" stroke-linejoin="round"/>')
    S.append(f'<polygon points="{p(90,41)} {p(97,22)} {p(108,41)}" fill="#C6C2BB" stroke="none"/>')
    # Near ear
    S.append(f'<polygon points="{p(57,42)} {p(67,15)} {p(83,42)}" fill="{cf}" stroke="{cs}" stroke-width="{CAT_SW}" stroke-linejoin="round"/>')
    S.append(f'<polygon points="{p(61,41)} {p(67,20)} {p(80,41)}" fill="#C6C2BB" stroke="none"/>')
    # Neck connector (no stroke, merges head+body)
    S.append(f'<polygon points="{p(72,82)} {p(108,82)} {p(112,122)} {p(76,122)}" fill="{cf}" stroke="none"/>')
    # Head
    S.append(f'<ellipse cx="{ox+80}" cy="{oy+64}" rx="29" ry="25" fill="{cf}" stroke="{cs}" stroke-width="{CAT_SW}"/>')
    # Tail (3 stacked paths: wide shadow, fill, thin outline)
    tp = f'M{ox+258},{oy+116} C{ox+278},{oy+92} {ox+286},{oy+60} {ox+268},{oy+43} C{ox+257},{oy+32} {ox+240},{oy+36} {ox+240},{oy+50}'
    S.append(f'<path d="{tp}" fill="none" stroke="{cs}" stroke-width="9" stroke-linecap="round" opacity="0.25"/>')
    S.append(f'<path d="{tp}" fill="none" stroke="{cf}" stroke-width="7" stroke-linecap="round"/>')
    S.append(f'<path d="{tp}" fill="none" stroke="{cs}" stroke-width="{CAT_SW}" stroke-linecap="round"/>')
    # Legs: far then near (near drawn on top)
    for lx,ly,lw,lh in [(216,150,13,62),(110,150,13,60),(196,148,14,64),(88,148,14,64)]:
        S.append(f'<rect x="{ox+lx}" y="{oy+ly}" width="{lw}" height="{lh}" rx="5" fill="{cf}" stroke="{cs}" stroke-width="{CAT_SW}"/>')
    # Paws
    for px,py,prx,pry in [(223,212,8,4),(116,211,7,4),(203,212,9,4),(95,212,9,4)]:
        S.append(f'<ellipse cx="{ox+px}" cy="{oy+py}" rx="{prx}" ry="{pry}" fill="{cf}" stroke="{cs}" stroke-width="{CAT_SW}"/>')
    # Nose triangle
    S.append(f'<polygon points="{p(105,66)} {p(111,66)} {p(108,70)}" fill="#998888" stroke="none"/>')
    return "\n      ".join(S)


def skel_svg(kps, skel, ox, oy):
    S = []
    for i1,i2,grp in skel:
        x1,y1 = kps[i1][0], kps[i1][1]
        x2,y2 = kps[i2][0], kps[i2][1]
        if x1 is None or x2 is None: continue
        col = COLORS[grp]
        S.append(f'<line x1="{ox+x1}" y1="{oy+y1}" x2="{ox+x2}" y2="{oy+y2}" '
                 f'stroke="{col}" stroke-width="1.8" stroke-linecap="round" opacity="0.88"/>')
    return "\n      ".join(S)


def kp_svg(kps, ox, oy, r=4.5):
    S = []
    for i,(x,y,name,grp) in enumerate(kps):
        if x is None: continue
        col = COLORS[grp]
        S.append(f'<circle cx="{ox+x}" cy="{oy+y}" r="{r}" fill="{col}" stroke="white" stroke-width="1.2"/>')
        fs = 7 if i < 10 else 6.5
        S.append(f'<text x="{ox+x}" y="{oy+y}" text-anchor="middle" dominant-baseline="central" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="{fs}" font-weight="bold" fill="white">{i}</text>')
    return "\n      ".join(S)


def legend_vit(x0, y0):
    """Single-column legend for ViTPose."""
    S = []
    # Column header
    for txt, cx, anchor in [("Idx","",""),("+14  Keypoint","",""),("+195  Region","","")]:
        pass
    S.append(f'<text x="{x0+8}" y="{y0}" font-family="Helvetica,Arial,sans-serif" font-size="9.5" font-weight="bold" fill="#555">Idx   Keypoint                    Region</text>')
    S.append(f'<line x1="{x0}" y1="{y0+4}" x2="{x0+365}" y2="{y0+4}" stroke="#bbb" stroke-width="0.5"/>')
    RH = 16
    for i,(x,y,name,grp) in enumerate(VIT):
        col = COLORS[grp]
        ry = y0 + 10 + i*RH
        if i%2==0:
            S.append(f'<rect x="{x0-2}" y="{ry-8}" width="370" height="{RH}" fill="#F6F6F4" rx="2"/>')
        S.append(f'<circle cx="{x0+8}" cy="{ry}" r="5" fill="{col}"/>')
        S.append(f'<text x="{x0+8}" y="{ry}" text-anchor="middle" dominant-baseline="central" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="6.5" font-weight="bold" fill="white">{i}</text>')
        S.append(f'<text x="{x0+18}" y="{ry}" dominant-baseline="central" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#333">{name}</text>')
        S.append(f'<text x="{x0+210}" y="{ry}" dominant-baseline="central" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="9" fill="{col}">{grp}</text>')
    return "\n      ".join(S)


def legend_sa(x0, y0):
    """Two-column legend for SuperAnimal."""
    S = []
    # Headers for both columns
    for ci, cx in enumerate([x0, x0+385]):
        S.append(f'<text x="{cx+8}" y="{y0}" font-family="Helvetica,Arial,sans-serif" font-size="9.5" font-weight="bold" fill="#555">Idx   Keypoint                 Region</text>')
        S.append(f'<line x1="{cx}" y1="{y0+4}" x2="{cx+378}" y2="{y0+4}" stroke="#bbb" stroke-width="0.5"/>')
    
    RH = 15
    for i,(x,y,name,grp) in enumerate(SA):
        col = COLORS[grp]
        col_idx = i // 20  # 0 or 1
        row_i   = i % 20
        cx = x0 + col_idx * 385
        ry = y0 + 10 + row_i * RH
        if row_i % 2 == 0:
            S.append(f'<rect x="{cx-2}" y="{ry-7}" width="382" height="{RH}" fill="#F6F6F4" rx="2"/>')
        # NA style
        if x is None:
            S.append(f'<circle cx="{cx+8}" cy="{ry}" r="4.5" fill="none" stroke="{col}" stroke-width="1" stroke-dasharray="2,1"/>')
            S.append(f'<text x="{cx+18}" y="{ry}" dominant-baseline="central" '
                     f'font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#AAA">{name}</text>')
            S.append(f'<text x="{cx+200}" y="{ry}" dominant-baseline="central" '
                     f'font-family="Helvetica,Arial,sans-serif" font-size="8.5" fill="#CCC">N/A (cat)</text>')
        else:
            S.append(f'<circle cx="{cx+8}" cy="{ry}" r="4.5" fill="{col}"/>')
            S.append(f'<text x="{cx+8}" y="{ry}" text-anchor="middle" dominant-baseline="central" '
                     f'font-family="Helvetica,Arial,sans-serif" font-size="6.5" font-weight="bold" fill="white">{i}</text>')
            S.append(f'<text x="{cx+18}" y="{ry}" dominant-baseline="central" '
                     f'font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#333">{name}</text>')
            S.append(f'<text x="{cx+200}" y="{ry}" dominant-baseline="central" '
                     f'font-family="Helvetica,Arial,sans-serif" font-size="9" fill="{col}">{grp}</text>')
    return "\n      ".join(S)


def color_legend(x0, y0):
    """Color-coding legend row."""
    items = [
        ("head/face", "#E07040"),
        ("spine/tail", "#3E80C0"),
        ("neck/throat", "#208E78"),
        ("front limbs", "#3EA060"),
        ("hind limbs", "#9050B0"),
        ("body/other", "#909090"),
    ]
    S = []
    cx = x0
    for label, col in items:
        S.append(f'<circle cx="{cx+5}" cy="{y0}" r="4" fill="{col}"/>')
        S.append(f'<text x="{cx+13}" y="{y0}" dominant-baseline="central" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="9" fill="#444">{label}</text>')
        cx += 112
    return "\n      ".join(S)


# ===================== BUILD SVG =====================
W = 780
H = 825

# Offsets
CAT_OX_A, CAT_OY_A = 20, 68
CAT_OX_B, CAT_OY_B = 20, 445

LEG_X_A, LEG_Y_A = 348, 65
LEG_X_B, LEG_Y_B = 348, 440

svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     font-family="Helvetica Neue,Arial,sans-serif">
  <title>Cat pose keypoint schemas</title>
  <desc>Two panels showing ViTPose 17-joint (APT-36K) and SuperAnimal 39-point keypoint schemas for cat pose estimation.</desc>

  <!-- Background -->
  <rect width="{W}" height="{H}" fill="white"/>

  <!-- ═══════════════  PANEL A — ViTPose  ═══════════════ -->
  <rect x="4" y="8" width="{W-8}" height="365" rx="5"
        fill="none" stroke="#CCCCCC" stroke-width="0.8"/>

  <!-- Panel label -->
  <text x="16" y="30" font-size="13" font-weight="bold" fill="#222">(a)</text>
  <text x="36" y="30" font-size="12" font-weight="bold" fill="#222">ViTPose / APT-36K skeleton</text>
  <text x="36" y="43" font-size="10" fill="#666">n = 17 keypoints · checkpoint: vitpose-h-apt36k.pth</text>

  <!-- Cat silhouette A -->
  <g id="cat-sil-a">
    {cat_svg(CAT_OX_A, CAT_OY_A)}
  </g>

  <!-- Skeleton A -->
  <g id="skel-a" opacity="0.9">
    {skel_svg(VIT, VIT_SKEL, CAT_OX_A, CAT_OY_A)}
  </g>

  <!-- Keypoints A -->
  <g id="kp-a">
    {kp_svg(VIT, CAT_OX_A, CAT_OY_A, r=5)}
  </g>

  <!-- Legend A -->
  <g id="leg-a">
    {legend_vit(LEG_X_A, LEG_Y_A)}
  </g>

  <!-- ═══════════════  PANEL B — SuperAnimal  ═══════════════ -->
  <rect x="4" y="380" width="{W-8}" height="425" rx="5"
        fill="none" stroke="#CCCCCC" stroke-width="0.8"/>

  <text x="16" y="400" font-size="13" font-weight="bold" fill="#222">(b)</text>
  <text x="36" y="400" font-size="12" font-weight="bold" fill="#222">DeepLabCut SuperAnimal quadruped vocabulary</text>
  <text x="36" y="413" font-size="10" fill="#666">n = 39 keypoints · indices 0–38 · points 8,9,13,14 (antlers) absent in cats</text>

  <!-- Cat silhouette B -->
  <g id="cat-sil-b">
    {cat_svg(CAT_OX_B, CAT_OY_B)}
  </g>

  <!-- Skeleton B -->
  <g id="skel-b" opacity="0.9">
    {skel_svg(SA, SA_SKEL, CAT_OX_B, CAT_OY_B)}
  </g>

  <!-- Keypoints B -->
  <g id="kp-b">
    {kp_svg(SA, CAT_OX_B, CAT_OY_B, r=4.5)}
  </g>

  <!-- Legend B -->
  <g id="leg-b">
    {legend_sa(LEG_X_B, LEG_Y_B)}
  </g>

  <!-- ═══════════════  Color-coding legend (shared) ═══════════════ -->
  <line x1="8" y1="810" x2="{W-8}" y2="810" stroke="#DDDDDD" stroke-width="0.5"/>
  <text x="12" y="820" font-size="9" font-weight="bold" fill="#666">Color key: </text>
  <g transform="translate(68, 820)">
    {color_legend(0, 0)}
  </g>

</svg>'''

out = "cat_keypoints_figure.svg"
with open(out, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Written: {out}")
print(f"Size: {len(svg):,} bytes")