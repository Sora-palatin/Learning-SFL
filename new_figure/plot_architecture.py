"""
LENS-SFL 系统架构图 — matplotlib 绘制 (v4 — 左右横向布局, 工程化风格, 纯黑白灰)
布局: 左侧 Server (Controller + Deep Layers)
      右侧 Clients (Heterogeneous + Shallow Layers)
      中间通道: 契约分发(顶) / SFL训练(中) / 反馈闭环(底)
输出: lens_sfl_architecture.pdf / .png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.path import Path

OUTPUT_DIR = os.path.dirname(__file__)

# ── 灰阶颜色 ──
BK  = '#000000'
DK  = '#333333'
MD  = '#666666'
LT  = '#999999'
VLT = '#CCCCCC'
BG  = '#F0F0F0'
WH  = '#FFFFFF'

LW = 1.5  # 统一线宽


# ────────────────────────────────────────────────
#  工具函数
# ────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, label=None, sublabel=None,
             facecolor=WH, edgecolor=BK, lw=LW, fontsize=16,
             sublabel_fontsize=16, zorder=5, round_pad=0.02,
             label_color=DK, sublabel_color=MD):
    """绘制圆角矩形方框 + 居中标签"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={round_pad}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         lw=lw, zorder=zorder)
    ax.add_patch(box)
    cx, cy = x + w/2, y + h/2
    if label and sublabel:
        ax.text(cx, cy + h*0.10, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=label_color, zorder=zorder+1)
        ax.text(cx, cy - h*0.15, sublabel, ha='center', va='center',
                fontsize=sublabel_fontsize, color=sublabel_color,
                style='italic', zorder=zorder+1)
    elif label:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=label_color, zorder=zorder+1)
    return box


def draw_arrow(ax, start, end, lw=LW, style='-',
               arrowstyle='->', mutation_scale=14, zorder=10,
               connectionstyle='arc3,rad=0', label=None, label_pos=None,
               label_fontsize=16, bidirectional=False, label_bg=True,
               label_offset=(0, 0)):
    """绘制黑色箭头连线"""
    astyle = '<->' if bidirectional else arrowstyle
    arrow = FancyArrowPatch(
        start, end, arrowstyle=astyle, mutation_scale=mutation_scale,
        color=BK, lw=lw, linestyle=style,
        connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(arrow)
    if label:
        if label_pos is None:
            label_pos = ((start[0]+end[0])/2 + label_offset[0],
                         (start[1]+end[1])/2 + label_offset[1])
        bbox_props = dict(boxstyle='round,pad=0.12', facecolor=WH,
                          edgecolor='none', alpha=1.0) if label_bg else None
        ax.text(label_pos[0], label_pos[1], label, ha='center', va='center',
                fontsize=label_fontsize, color=DK, fontweight='bold',
                zorder=zorder+1, bbox=bbox_props)


def draw_step_label(ax, cx, cy, text, fontsize=32):
    """直接在线上标注序号文字 (无背景)"""
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=BK, zorder=26)


def _draw_device_icons(ax, dev_centers, dev_w, dev_h):
    """在三个设备框内绘制简笔图标: 芯片(小), 手机(中), 主机(大), 彩色"""
    # ── IoT: 芯片图标 (最小) ──
    cx, cy = dev_centers[0]
    cy += 0.15
    s = 0.20  # 芯片半边长 (小)
    chip = FancyBboxPatch((cx - s, cy - s), 2*s, 2*s,
                           boxstyle='round,pad=0.02',
                           facecolor='#4CAF50', edgecolor='#2E7D32',
                           lw=1.2, zorder=6)
    ax.add_patch(chip)
    # 芯片内部方块
    inner_s = s * 0.5
    chip_inner = plt.Rectangle((cx - inner_s, cy - inner_s),
                                2*inner_s, 2*inner_s,
                                facecolor='#388E3C', edgecolor='#1B5E20',
                                lw=0.6, zorder=7)
    ax.add_patch(chip_inner)
    # 引脚 (上下左右各3根)
    pin_len = 0.08
    for k in range(3):
        offset = -0.10 + k * 0.10
        ax.plot([cx + offset, cx + offset], [cy + s, cy + s + pin_len],
                color='#2E7D32', lw=0.8, zorder=6)
        ax.plot([cx + offset, cx + offset], [cy - s, cy - s - pin_len],
                color='#2E7D32', lw=0.8, zorder=6)
        ax.plot([cx - s, cx - s - pin_len], [cy + offset, cy + offset],
                color='#2E7D32', lw=0.8, zorder=6)
        ax.plot([cx + s, cx + s + pin_len], [cy + offset, cy + offset],
                color='#2E7D32', lw=0.8, zorder=6)

    # ── Mobile: 手机图标 (中等) ──
    cx2, cy2 = dev_centers[1]
    cy2 += 0.15
    pw, ph = 0.38, 0.65  # 手机宽高 (中)
    phone = FancyBboxPatch((cx2 - pw/2, cy2 - ph/2), pw, ph,
                            boxstyle='round,pad=0.04',
                            facecolor='#42A5F5', edgecolor='#1565C0',
                            lw=1.2, zorder=6)
    ax.add_patch(phone)
    # 屏幕
    scr_margin = 0.06
    screen = plt.Rectangle((cx2 - pw/2 + scr_margin, cy2 - ph/2 + 0.14),
                             pw - 2*scr_margin, ph - 0.26,
                             facecolor='#BBDEFB', edgecolor='#1565C0',
                             lw=0.5, zorder=7)
    ax.add_patch(screen)
    # Home 键
    home_btn = plt.Circle((cx2, cy2 - ph/2 + 0.07), 0.04,
                           facecolor='#1565C0', edgecolor='#0D47A1',
                           lw=0.5, zorder=7)
    ax.add_patch(home_btn)

    # ── PC: 主机图标 (最大) ──
    cx3, cy3 = dev_centers[2]
    cy3 += 0.15
    bw, bh = 0.65, 0.85  # 主机箱宽高 (大)
    case = FancyBboxPatch((cx3 - bw/2, cy3 - bh/2), bw, bh,
                           boxstyle='round,pad=0.02',
                           facecolor='#78909C', edgecolor='#37474F',
                           lw=1.2, zorder=6)
    ax.add_patch(case)
    # 光驱槽
    slot_y = cy3 + bh/2 - 0.18
    slot = plt.Rectangle((cx3 - bw/2 + 0.08, slot_y - 0.10),
                          bw - 0.16, 0.10,
                          facecolor='#546E7A', edgecolor='#37474F',
                          lw=0.6, zorder=7)
    ax.add_patch(slot)
    # 电源按钮
    btn = plt.Circle((cx3, cy3 - bh/2 + 0.18), 0.07,
                      facecolor='#4CAF50', edgecolor='#2E7D32',
                      lw=0.8, zorder=7)
    ax.add_patch(btn)
    # 散热格栅
    for gi in range(5):
        gy = cy3 - 0.10 + gi * 0.10
        ax.plot([cx3 - bw/2 + 0.08, cx3 + bw/2 - 0.08], [gy, gy],
                color='#546E7A', lw=0.6, zorder=7)


def draw_layer_stack(ax, x, y, w, h, n_layers, labels=None,
                     gray_start=0.30, gray_end=0.75, zorder=5):
    """绘制堆叠网络层 (灰度渐变)
    x,y = 左下角; w,h = 总宽高
    返回 (top_y, bot_y, cx) 用于连线
    """
    gap = 0.04
    layer_h = (h - (n_layers - 1) * gap) / n_layers
    cx = x + w / 2
    for i in range(n_layers):
        ly = y + h - (i + 1) * layer_h - i * gap
        t = i / max(n_layers - 1, 1)
        g = gray_start + t * (gray_end - gray_start)
        gc = int(g * 255)
        col = f'#{gc:02X}{gc:02X}{gc:02X}'
        rect = FancyBboxPatch((x, ly), w, layer_h,
                               boxstyle="round,pad=0.008",
                               facecolor=col, edgecolor=BK,
                               lw=0.6, zorder=zorder)
        ax.add_patch(rect)
        if labels and i < len(labels):
            tc = WH if g < 0.55 else DK
            ax.text(cx, ly + layer_h/2, labels[i],
                    ha='center', va='center', fontsize=16,
                    color=tc, fontweight='bold', zorder=zorder+1)
    return (y + h, y, cx)


# ────────────────────────────────────────────────
#  主函数
# ────────────────────────────────────────────────

def main():
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))
    ax.set_xlim(-0.5, 18.0)
    ax.set_ylim(-1.5, 9.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ================================================================
    #  坐标系规划 (左右横向布局)
    #  左侧 Server: x=[0.0, 5.0]
    #  中间通道:     x=[5.0, 10.0]
    #  右侧 Client:  x=[10.0, 15.0]
    # ================================================================

    # ============================================================
    #  Step 1: Server 侧 (左侧大框)
    # ============================================================
    srv_x, srv_y, srv_w, srv_h = 0.2, 0.0, 4.6, 7.8
    srv_outer = FancyBboxPatch((srv_x, srv_y), srv_w, srv_h,
                                boxstyle="round,pad=0.06",
                                facecolor='#E8EAF6', edgecolor=BK,
                                lw=2.0, zorder=1)
    ax.add_patch(srv_outer)
    ax.text(srv_x + srv_w/2, srv_y + srv_h + 0.25,
            'Cloud Server',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=BK, zorder=3)

    # ── 上部: Control Plane — LENS-UCB Controller ──
    ctrl_x = srv_x + 0.4
    ctrl_y = srv_y + srv_h - 2.8
    ctrl_w = srv_w - 0.8
    ctrl_h = 2.4
    draw_box(ax, ctrl_x, ctrl_y, ctrl_w, ctrl_h,
             facecolor='#C5CAE9', edgecolor=BK, lw=LW, zorder=3)
    ax.text(ctrl_x + ctrl_w/2, ctrl_y + ctrl_h - 0.25,
            'LENS-UCB', ha='center', va='center',
            fontsize=16, fontweight='bold', color=BK, zorder=4)
    # ── 思考人头 + 向上气泡 + 循环曲线(气泡末端) 居中 ──
    ctrl_cx = ctrl_x + ctrl_w / 2

    # 人头 (底部居中)
    face_r = 0.28
    face_cx = ctrl_cx - 0.15
    face_cy = ctrl_y + 0.45
    face = plt.Circle((face_cx, face_cy), face_r,
                       facecolor='#FFF9C4', edgecolor=BK, lw=1.2, zorder=5)
    ax.add_patch(face)
    # 左眼
    ax.plot(face_cx - 0.09, face_cy + 0.05, 'o', color=BK, markersize=2.5, zorder=6)
    # 右眼
    ax.plot(face_cx + 0.09, face_cy + 0.07, 'o', color=BK, markersize=2.5, zorder=6)
    # 左眉 (微皱)
    ax.plot([face_cx - 0.13, face_cx - 0.04], [face_cy + 0.14, face_cy + 0.16],
            color=BK, lw=0.9, zorder=6)
    # 右眉 (微抬)
    ax.plot([face_cx + 0.04, face_cx + 0.13], [face_cy + 0.16, face_cy + 0.14],
            color=BK, lw=0.9, zorder=6)
    # 思考嘴 (偏右短横)
    ax.plot([face_cx + 0.02, face_cx + 0.12], [face_cy - 0.10, face_cy - 0.08],
            color=BK, lw=1.0, zorder=6)

    # 气泡: 从人头右上方向上递增
    bubble_data = [
        (face_cx + 0.22, face_cy + face_r + 0.10, 0.04),
        (face_cx + 0.30, face_cy + face_r + 0.24, 0.055),
        (face_cx + 0.35, face_cy + face_r + 0.40, 0.07),
    ]
    for bx, by, br in bubble_data:
        bubble = plt.Circle((bx, by), br,
                             facecolor=WH, edgecolor=BK, lw=0.8, zorder=5)
        ax.add_patch(bubble)

    # 循环曲线 (气泡末端上方, 紧贴最后一个气泡)
    loop_r = 0.28
    loop_cx = bubble_data[-1][0] + 0.03
    loop_cy = bubble_data[-1][1] + bubble_data[-1][2] + loop_r * 0.8 + 0.04
    loop_arc = Arc((loop_cx, loop_cy), loop_r*2, loop_r*1.5,
                   angle=0, theta1=30, theta2=330,
                   color=BK, lw=1.8, zorder=5)
    ax.add_patch(loop_arc)
    a_ang = np.radians(30)
    a_x = loop_cx + loop_r * np.cos(a_ang)
    a_y = loop_cy + loop_r * 0.75 * np.sin(a_ang)
    ax.annotate('', xy=(a_x + 0.06, a_y - 0.08),
                xytext=(a_x, a_y),
                arrowprops=dict(arrowstyle='->', color=BK, lw=1.8),
                zorder=5)

    # 序号 ① 在人头左侧
    draw_step_label(ax, face_cx - face_r - 0.22, face_cy, '①')

    # ── 下部: Compute Plane — Server-Side Model ──
    comp_x = srv_x + 0.4
    comp_y = srv_y + 0.3
    comp_w = srv_w - 0.8
    comp_h = ctrl_y - comp_y - 0.3
    draw_box(ax, comp_x, comp_y, comp_w, comp_h,
             facecolor='#BBDEFB', edgecolor=BK, lw=LW, zorder=3)
    ax.text(comp_x + comp_w/2, comp_y + comp_h - 0.25,
            'Server-Side Model',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=BK, zorder=4)

    # 内部纵向矩形 (竖条, 低于标题文字)
    n_inner = 6
    inner_pad_x = 0.15
    inner_pad_y_top = 0.55
    inner_pad_y_bot = 0.15
    inner_area_x = comp_x + inner_pad_x
    inner_area_w = comp_w - 2 * inner_pad_x
    inner_area_y = comp_y + inner_pad_y_bot
    inner_area_h = comp_h - inner_pad_y_top - inner_pad_y_bot
    v_gap = 0.06
    v_col_w = (inner_area_w - (n_inner - 1) * v_gap) / n_inner
    # Server竖条: 蓝色渐变, 左深右浅 (靠近图边缘深, 靠近中间浅)
    for i in range(n_inner):
        vx = inner_area_x + i * (v_col_w + v_gap)
        t = i / max(n_inner - 1, 1)
        # 从深蓝到浅蓝 (左→右, 即边缘→中间)
        r = int(25 + t * (144 - 25))
        g = int(118 + t * (202 - 118))
        b = int(210 + t * (249 - 210))
        col = f'#{r:02X}{g:02X}{b:02X}'
        rect = FancyBboxPatch((vx, inner_area_y), v_col_w, inner_area_h,
                               boxstyle='round,pad=0.008',
                               facecolor=col, edgecolor=BK,
                               lw=0.5, zorder=4)
        ax.add_patch(rect)

    # Server 侧关键坐标
    srv_right = srv_x + srv_w
    ctrl_right = ctrl_x + ctrl_w
    ctrl_cy = ctrl_y + ctrl_h / 2
    comp_cy = comp_y + comp_h / 2

    # ============================================================
    #  Step 2: Client 侧 — 分为上方设备框 + 下方模型框
    # ============================================================

    # ── 上方: Heterogeneous Clients 独立框 ──
    hc_x = 11.5
    hc_y = ctrl_y          # 与 Controller 同高
    hc_w = 5.4
    hc_h = ctrl_h          # 与 Controller 同高
    hc_outer = FancyBboxPatch((hc_x, hc_y), hc_w, hc_h,
                               boxstyle="round,pad=0.06",
                               facecolor='#FFF3E0', edgecolor=BK,
                               lw=2.0, zorder=1)
    ax.add_patch(hc_outer)
    ax.text(hc_x + hc_w/2, hc_y + hc_h + 0.20,
            'Heterogeneous Clients',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=BK, zorder=3)

    # 3个设备水平排列
    dev_w = (hc_w - 0.8) / 3
    dev_h_inner = hc_h - 0.6
    dev_labels = ['IoT\nDevice', 'Mobile', 'PC /\nWorkstation']
    dev_centers = []
    for i, lbl in enumerate(dev_labels):
        dx = hc_x + 0.2 + i * (dev_w + 0.1)
        dy = hc_y + 0.3
        dev_bg_colors = ['#E8F5E9', '#E3F2FD', '#ECEFF1']
        draw_box(ax, dx, dy, dev_w, dev_h_inner,
                 facecolor=dev_bg_colors[i], edgecolor=BK,
                 lw=LW, fontsize=16, zorder=3)
        dcx, dcy = dx + dev_w/2, dy + dev_h_inner/2
        dev_centers.append((dcx, dcy))
        # 标签放在下方
        ax.text(dcx, dy + 0.18, lbl, ha='center', va='center',
                fontsize=12, fontweight='bold', color=DK, zorder=4)

    # 绘制设备图标
    _draw_device_icons(ax, dev_centers, dev_w, dev_h_inner)

    hc_left = hc_x
    hc_right = hc_x + hc_w
    hc_bot = hc_y
    hc_cx = hc_x + hc_w / 2

    # ── 下方: Client-Side Model 大模块 (与 Server-Side Model 同高) ──
    cli_mod_x = 11.5
    cli_mod_y = comp_y       # 与 Server-Side Model 底部对齐
    cli_mod_w = 5.4
    cli_mod_h = comp_h       # 与 Server-Side Model 同高
    draw_box(ax, cli_mod_x, cli_mod_y, cli_mod_w, cli_mod_h,
             facecolor='#BBDEFB', edgecolor=BK, lw=LW, zorder=3)
    ax.text(cli_mod_x + cli_mod_w/2, cli_mod_y + cli_mod_h - 0.25,
            'Client-Side Model',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=BK, zorder=4)

    # 内部纵向矩形 (竖条, 低于标题文字)
    cli_inner_pad_x = 0.15
    cli_inner_pad_y_top = 0.55
    cli_inner_pad_y_bot = 0.15
    cli_area_x = cli_mod_x + cli_inner_pad_x
    cli_area_w = cli_mod_w - 2 * cli_inner_pad_x
    cli_area_y = cli_mod_y + cli_inner_pad_y_bot
    cli_area_h = cli_mod_h - cli_inner_pad_y_top - cli_inner_pad_y_bot
    cli_v_gap = 0.06
    cli_v_col_w = (cli_area_w - (n_inner - 1) * cli_v_gap) / n_inner
    # Client竖条: 蓝色渐变, 左浅右深 (靠近中间浅, 靠近边缘深)
    for i in range(n_inner):
        vx = cli_area_x + i * (cli_v_col_w + cli_v_gap)
        t = i / max(n_inner - 1, 1)
        # 从浅蓝到深蓝 (左→右, 即中间→边缘)
        r = int(144 + t * (25 - 144))
        g = int(202 + t * (118 - 202))
        b = int(249 + t * (210 - 249))
        col = f'#{r:02X}{g:02X}{b:02X}'
        rect = FancyBboxPatch((vx, cli_area_y), cli_v_col_w, cli_area_h,
                               boxstyle='round,pad=0.008',
                               facecolor=col, edgecolor=BK,
                               lw=0.5, zorder=4)
        ax.add_patch(rect)

    # Heterogeneous Clients → Client-Side Model 连接箭头
    draw_arrow(ax, (hc_cx, hc_bot),
               (cli_mod_x + cli_mod_w/2, cli_mod_y + cli_mod_h),
               lw=LW, zorder=8)

    # Client 侧关键坐标
    cli_left = hc_x
    cli_mod_left = cli_mod_x
    cli_mod_right = cli_mod_x + cli_mod_w

    # ============================================================
    #  Step 3: 契约分发流 (顶部数据流)
    # ============================================================
    # Contract Menu 表格 (中间偏上)
    menu_x = 6.0
    menu_y = 6.0
    menu_w = 4.0
    menu_h = 1.5
    draw_box(ax, menu_x, menu_y, menu_w, menu_h,
             facecolor='#FFF3E0', edgecolor='#E65100', lw=LW, zorder=5)
    ax.text(menu_x + menu_w/2, menu_y + menu_h - 0.22,
            'Contract Menu $\\mathcal{M}$',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=BK, zorder=6)

    # 表格内容
    table_items = [
        '$(v_1, R_1)$',
        '$\\cdots$',
        '$(v_K, R_K)$',
    ]
    item_w = (menu_w - 0.2) / len(table_items)
    for i, txt in enumerate(table_items):
        ix = menu_x + 0.1 + i * item_w
        iy = menu_y + 0.15
        ih = menu_h - 0.55
        rect = plt.Rectangle((ix, iy), item_w - 0.05, ih,
                              facecolor='#FFE0B2', edgecolor='#F57C00', lw=0.5, zorder=6)
        ax.add_patch(rect)
        ax.text(ix + (item_w - 0.05)/2, iy + ih/2, txt,
                ha='center', va='center', fontsize=20, color=DK, zorder=7)

    # 箭头: Controller → Menu (②)
    draw_arrow(ax, (srv_right, ctrl_y + ctrl_h - 0.4),
               (menu_x, menu_y + menu_h/2),
               lw=LW, connectionstyle='arc3,rad=0.10', zorder=10)
    # 序号 ②
    mid2_x = (srv_right + menu_x) / 2
    mid2_y = (ctrl_y + ctrl_h - 0.4 + menu_y + menu_h/2) / 2 + 0.25
    draw_step_label(ax, mid2_x, mid2_y, '②')

    # 箭头: Menu → Heterogeneous Clients (③)
    draw_arrow(ax, (menu_x + menu_w, menu_y + menu_h/2),
               (hc_left, hc_y + hc_h/2),
               lw=LW, connectionstyle='arc3,rad=-0.10', zorder=10)
    # 序号 ③
    mid3_x = (menu_x + menu_w + hc_left) / 2
    mid3_y = (menu_y + menu_h/2 + hc_y + hc_h/2) / 2 + 0.25
    draw_step_label(ax, mid3_x, mid3_y, '③')

    # ============================================================
    #  Step 4: SFL 训练流 (中间水平双向箭头)
    # ============================================================
    sfl_y_upper = comp_cy + 0.4   # Forward
    sfl_y_lower = comp_cy - 0.4   # Backward

    # Forward: Client-Side Model → Server-Side Model (右→左, Activations)
    draw_arrow(ax, (cli_mod_left, sfl_y_upper),
               (srv_right, sfl_y_upper),
               lw=LW, zorder=10)
    ax.text((cli_mod_left + srv_right) / 2, sfl_y_upper + 0.25,
            'Forward (Activations)',
            ha='center', va='bottom', fontsize=16, fontweight='bold',
            color=DK, zorder=11,
            bbox=dict(boxstyle='round,pad=0.10', facecolor=WH,
                      edgecolor='none', alpha=0.9))

    # Backward: Server-Side Model → Client-Side Model (左→右, Gradients)
    draw_arrow(ax, (srv_right, sfl_y_lower),
               (cli_mod_left, sfl_y_lower),
               lw=LW, zorder=10)
    ax.text((cli_mod_left + srv_right) / 2, sfl_y_lower - 0.25,
            'Backward (Gradients)',
            ha='center', va='top', fontsize=16, fontweight='bold',
            color=DK, zorder=11,
            bbox=dict(boxstyle='round,pad=0.10', facecolor=WH,
                      edgecolor='none', alpha=0.9))

    # Cut Layer 标注 (中间虚线)
    cut_x = (srv_right + cli_mod_left) / 2
    ax.plot([cut_x, cut_x], [sfl_y_lower - 0.8, sfl_y_upper + 0.8],
            color='#D32F2F', lw=1.5, ls='--', zorder=8)
    # Cut Layer 标注放到下方
    ax.text(cut_x, sfl_y_lower - 1.0, 'Cut Layer $v^*$',
            ha='center', va='top', fontsize=16, fontweight='bold',
            color=BK, zorder=9,
            bbox=dict(boxstyle='round,pad=0.08', facecolor=WH,
                      edgecolor=BK, lw=0.5))

    # 序号 ④ (Forward和Backward中间, 偏右避开虚线)
    draw_step_label(ax, cut_x + 0.8, comp_cy, '④')

    # ============================================================
    #  Step 5: 反馈闭环 (柔和曲线穿过中间空白 → Server外壳靠上)
    # ============================================================
    # 起点: Client-Side Model 左外壳顶部
    fb_start = (cli_mod_left, cli_mod_y + cli_mod_h)
    # 终点: Server 外壳右侧, Controller 下方
    fb_end = (srv_right, ctrl_y)
    # 用 arc3 曲线柔和穿过中间空白区域 (负弧度向上凸)
    draw_arrow(ax, fb_start, fb_end,
               lw=LW, connectionstyle='arc3,rad=-0.12', zorder=10)

    # 标注 ⑤ (贴在反馈曲线上)
    fb_mid_x = (fb_start[0] + fb_end[0]) / 2
    fb_mid_y = (fb_start[1] + fb_end[1]) / 2
    chord_len = ((fb_start[0]-fb_end[0])**2 + (fb_start[1]-fb_end[1])**2)**0.5
    arc_offset = 0.12 * chord_len / 2
    draw_step_label(ax, fb_mid_x, fb_mid_y + arc_offset, '⑤')

    # ============================================================
    #  保存
    # ============================================================
    plt.tight_layout()
    out_pdf = os.path.join(OUTPUT_DIR, 'lens_sfl_architecture.pdf')
    out_png = os.path.join(OUTPUT_DIR, 'lens_sfl_architecture.png')
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'已保存: {out_pdf}')
    print(f'已保存: {out_png}')


if __name__ == '__main__':
    main()
