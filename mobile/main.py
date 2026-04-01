"""
Mine Tracker — Flet Mobile Dashboard
Consome os dados de mine-tracker/data/08_reporting/*.json
"""

import flet as ft
import json
import os
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

# ── Cores por nível ──────────────────────────────────────────────────────────
LEVEL_COLORS = {
    "alto":    ft.Colors.RED_400,
    "médio":   ft.Colors.AMBER_400,
    "baixo":   ft.Colors.GREEN_400,
    "crítico": ft.Colors.RED_700,
}
LEVEL_ICONS = {
    "alto":    ft.Icons.WARNING_ROUNDED,
    "médio":   ft.Icons.TRENDING_FLAT,
    "baixo":   ft.Icons.CHECK_CIRCLE_ROUNDED,
    "crítico": ft.Icons.ERROR_ROUNDED,
}


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_metricas() -> dict:
    path = os.path.join(DATA_DIR, 'metricas.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def build_summary() -> dict:
    """Pré-processa report_inference.json e retorna resumo compacto."""
    path = os.path.join(DATA_DIR, 'report.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.loads(f.read().replace(': NaN', ': null'))
    except Exception:
        return {}

    summary_clusters = []
    for cluster in raw.get('clusters', []):
        instances = cluster.get('instances', [])

        hourly: dict = defaultdict(lambda: {'playerCount': [], 'media_movel_10': [], 'target_24h': []})
        last_instance: dict = {}
        for inst in instances:
            h = int(inst.get('hora', 0) or 0)
            pc = inst.get('playerCount')
            if pc is not None:
                hourly[h]['playerCount'].append(pc)
            mm = inst.get('media_movel_10')
            if mm is not None:
                hourly[h]['media_movel_10'].append(mm)
            t24 = inst.get('target_24h')
            if t24 is not None:
                hourly[h]['target_24h'].append(t24)
            last_instance = inst

        hourly_summary = {}
        for h, vals in sorted(hourly.items()):
            hourly_summary[h] = {
                'avg_playerCount': round(sum(vals['playerCount']) / len(vals['playerCount']), 1) if vals['playerCount'] else None,
                'avg_target_24h': round(sum(vals['target_24h']) / len(vals['target_24h']), 1) if vals['target_24h'] else None,
                'count': len(vals['playerCount']),
            }

        first = instances[0] if instances else {}
        pc_first = first.get('playerCount')
        pc_last = last_instance.get('playerCount')
        if pc_first and pc_last and pc_first != 0:
            variacao_pct = round(((pc_last - pc_first) / pc_first) * 100, 2)
        else:
            variacao_pct = 0

        summary_clusters.append({
            'domain': cluster.get('domain'),
            'cluster_id': cluster.get('cluster_id'),
            'baseline_prediction': cluster.get('baseline_prediction'),
            'level': cluster.get('level'),
            'action': cluster.get('action'),
            'server_mean': last_instance.get('server_mean'),
            'media_movel_10': last_instance.get('media_movel_10'),
            'last_timestamp': last_instance.get('timestamp'),
            'variacao_pct': variacao_pct,
            'hourly': hourly_summary,
        })

    return {
        'legend': raw.get('legend', {}),
        'clusters': summary_clusters,
        'ranking': raw.get('ranking', []),
        'metricas': load_metricas(),
    }


# ── UI Helpers ───────────────────────────────────────────────────────────────
def _badge(level: str) -> ft.Container:
    color = LEVEL_COLORS.get(level, ft.Colors.GREY_400)
    return ft.Container(
        content=ft.Text(level.upper(), size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        bgcolor=color,
        border_radius=12,
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
    )


def _stat_tile(label: str, value, icon_name=None, color=None) -> ft.Container:
    children = []
    if icon_name:
        children.append(ft.Icon(icon_name, size=18, color=color))
    children.append(ft.Text(str(value), size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE))
    children.append(ft.Text(label, size=11, color=ft.Colors.WHITE54))
    return ft.Container(
        content=ft.Column(children, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
        bgcolor=ft.Colors.WHITE10,
        border_radius=12,
        padding=14,
        expand=True,
    )


# ── App principal ────────────────────────────────────────────────────────────
def main(page: ft.Page):
    page.title = "Mine Tracker"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.bgcolor = "#0d1117"
    page.theme = ft.Theme(font_family="Inter")

    # ── Loading ─────────────────────────────────────────────────────────────
    loading = ft.Container(
        content=ft.Column(
            [
                ft.ProgressRing(width=48, height=48, color=ft.Colors.BLUE_400),
                ft.Text("Carregando dados…", size=16, color=ft.Colors.WHITE54),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        expand=True,
    )
    page.add(loading)
    page.update()

    data = build_summary()
    clusters = data.get('clusters', [])
    ranking = data.get('ranking', [])
    metricas = data.get('metricas', {})

    page.clean()

    # =====================================================================
    # TAB 1 — Ranking
    # =====================================================================
    def build_ranking_tab():
        rows = []
        for r in sorted(ranking, key=lambda x: x.get('posicao', 99)):
            lvl = r.get('level', '')
            color = LEVEL_COLORS.get(lvl, ft.Colors.GREY_400)
            pos = r.get('posicao', '?')
            rows.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(
                                content=ft.Text(f"#{pos}", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                width=42,
                            ),
                            ft.Container(width=4, height=40, bgcolor=color, border_radius=2),
                            ft.Column(
                                [
                                    ft.Text(r.get('domain', ''), size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                                    ft.Text(f"Previsão 24h: {r.get('prediction', 0):,} jogadores", size=12, color=ft.Colors.WHITE54),
                                ],
                                spacing=2,
                                expand=True,
                            ),
                            _badge(lvl),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=10,
                    ),
                    bgcolor=ft.Colors.WHITE10,
                    border_radius=12,
                    padding=ft.Padding.symmetric(horizontal=14, vertical=10),
                    margin=ft.Margin.only(bottom=8),
                )
            )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Ranking de Servidores", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    ft.Text("Ordenado pela previsão de jogadores nas próximas 24h", size=12, color=ft.Colors.WHITE54),
                    ft.Divider(height=12, color=ft.Colors.TRANSPARENT),
                    *rows,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=0,
            ),
            padding=20,
            expand=True,
        )

    # =====================================================================
    # TAB 2 — Servidores (cards expansíveis)
    # =====================================================================
    def build_servers_tab():
        panels = []
        for c in clusters:
            lvl = c.get('level', '')
            color = LEVEL_COLORS.get(lvl, ft.Colors.GREY_400)
            icon = LEVEL_ICONS.get(lvl, ft.Icons.HELP)

            var = c.get('variacao_pct', 0)
            var_color = ft.Colors.GREEN_400 if var >= 0 else ft.Colors.RED_400
            var_icon = ft.Icons.TRENDING_UP if var > 0 else (ft.Icons.TRENDING_DOWN if var < 0 else ft.Icons.TRENDING_FLAT)

            server_mean_val = c.get('server_mean')
            server_mean_fmt = f"{server_mean_val:,.0f}" if server_mean_val else "N/A"

            hourly = c.get('hourly', {})
            peak_hour = max(hourly, key=lambda h: hourly[h].get('avg_playerCount') or 0, default=None) if hourly else None
            peak_players = hourly[peak_hour]['avg_playerCount'] if peak_hour is not None else None

            panel = ft.ExpansionTile(
                leading=ft.Icon(icon, color=color, size=26),
                title=ft.Text(c.get('domain', ''), size=15, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                subtitle=ft.Row(
                    [
                        _badge(lvl),
                        ft.Text(f"Previsão: {c.get('baseline_prediction', 0):,}", size=12, color=ft.Colors.WHITE54),
                    ],
                    spacing=8,
                ),
                bgcolor=ft.Colors.WHITE10,
                collapsed_bgcolor=ft.Colors.WHITE10,
                shape=ft.RoundedRectangleBorder(radius=12),
                controls=[
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        _stat_tile("Média Server", server_mean_fmt, ft.Icons.EQUALIZER, ft.Colors.BLUE_400),
                                        _stat_tile("Variação", f"{var:+.1f}%", var_icon, var_color),
                                    ],
                                    spacing=8,
                                ),
                                ft.Row(
                                    [
                                        _stat_tile("Pico (hora)", f"{peak_hour}h" if peak_hour is not None else "N/A", ft.Icons.ACCESS_TIME, ft.Colors.AMBER_400),
                                        _stat_tile("Pico (avg)", f"{peak_players:,.0f}" if peak_players else "N/A", ft.Icons.GROUPS, ft.Colors.PURPLE_400),
                                    ],
                                    spacing=8,
                                ),
                                ft.Container(
                                    content=ft.Text(
                                        f"💡 {c.get('action', '')}",
                                        size=13,
                                        color=ft.Colors.AMBER_200,
                                        italic=True,
                                    ),
                                    bgcolor=ft.Colors.AMBER_900,
                                    border_radius=10,
                                    padding=12,
                                    margin=ft.Margin.only(top=6),
                                ),
                                ft.Text(
                                    f"Última atualização: {c.get('last_timestamp', 'N/A')}",
                                    size=11,
                                    color=ft.Colors.WHITE38,
                                ),
                            ],
                            spacing=10,
                        ),
                        padding=ft.Padding.only(left=12, right=12, bottom=14, top=4),
                    ),
                ],
            )
            panels.append(ft.Container(content=panel, margin=ft.Margin.only(bottom=8)))

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Detalhes dos Servidores", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    ft.Text("Toque para expandir e ver métricas detalhadas", size=12, color=ft.Colors.WHITE54),
                    ft.Divider(height=12, color=ft.Colors.TRANSPARENT),
                    *panels,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=0,
            ),
            padding=20,
            expand=True,
        )

    # =====================================================================
    # TAB 3 — Métricas ML
    # =====================================================================
    def build_metrics_tab():
        if not metricas:
            return ft.Container(
                content=ft.Text("Nenhuma métrica disponível.", color=ft.Colors.WHITE54),
                padding=20,
                expand=True,
            )

        best_model = max(metricas, key=lambda m: metricas[m].get('cv_r2_mean', 0))
        cards = []
        for model_name, m in metricas.items():
            is_best = model_name == best_model
            border_color = ft.Colors.GREEN_400 if is_best else ft.Colors.WHITE10

            r2 = m.get('cv_r2_mean', 0)
            r2_bar_color = ft.Colors.GREEN_400 if r2 > 0.95 else ft.Colors.AMBER_400 if r2 > 0.8 else ft.Colors.RED_400

            card = ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.STAR if is_best else ft.Icons.ANALYTICS, color=ft.Colors.AMBER_400 if is_best else ft.Colors.BLUE_400, size=22),
                                ft.Text(model_name, size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE, expand=True),
                                ft.Container(
                                    content=ft.Text("MELHOR" if is_best else "", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
                                    bgcolor=ft.Colors.GREEN_400 if is_best else ft.Colors.TRANSPARENT,
                                    border_radius=8,
                                    padding=ft.Padding.symmetric(horizontal=8, vertical=3),
                                ),
                            ],
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Divider(height=6, color=ft.Colors.WHITE10),
                        ft.Column(
                            [
                                ft.Row([
                                    ft.Text("R² (CV)", size=12, color=ft.Colors.WHITE54, expand=True),
                                    ft.Text(f"{r2:.4f}", size=13, weight=ft.FontWeight.BOLD, color=r2_bar_color),
                                ]),
                                ft.ProgressBar(value=max(0.0, float(r2)), color=r2_bar_color, bgcolor=ft.Colors.WHITE10, bar_height=6, border_radius=3),
                            ],
                            spacing=4,
                        ),
                        ft.Row(
                            [
                                ft.Column(
                                    [
                                        ft.Text("MAE", size=11, color=ft.Colors.WHITE54),
                                        ft.Text(f"{m.get('cv_mae_mean', 0):,.0f}", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                                    ],
                                    expand=True,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                ft.Column(
                                    [
                                        ft.Text("RMSE", size=11, color=ft.Colors.WHITE54),
                                        ft.Text(f"{m.get('cv_rmse_mean', 0):,.0f}", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                                    ],
                                    expand=True,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                ft.Column(
                                    [
                                        ft.Text("R² Std", size=11, color=ft.Colors.WHITE54),
                                        ft.Text(f"±{m.get('cv_r2_std', 0):.4f}", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                                    ],
                                    expand=True,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                            ],
                            spacing=0,
                        ),
                    ],
                    spacing=8,
                ),
                bgcolor="#161b22",
                border=ft.border.all(2, border_color),
                border_radius=14,
                padding=16,
                margin=ft.Margin.only(bottom=10),
            )
            cards.append(card)

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Métricas dos Modelos", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    ft.Text("Comparação cross-validation dos modelos treinados", size=12, color=ft.Colors.WHITE54),
                    ft.Divider(height=12, color=ft.Colors.TRANSPARENT),
                    *cards,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=0,
            ),
            padding=20,
            expand=True,
        )

    # ── Navegação com NavigationBar ─────────────────────────────────────────
    views = [build_ranking_tab(), build_servers_tab(), build_metrics_tab()]
    content_area = ft.Container(content=views[0], expand=True)

    def on_nav_change(e):
        idx = e.control.selected_index
        content_area.content = views[idx]
        page.update()

    header = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.DIAMOND, color=ft.Colors.GREEN_400, size=28),
                ft.Text("Mine Tracker", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.only(left=20, right=20, top=44, bottom=8),
        bgcolor="#0d1117",
    )

    nav_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.LEADERBOARD, label="Ranking"),
            ft.NavigationBarDestination(icon=ft.Icons.DNS, label="Servidores"),
            ft.NavigationBarDestination(icon=ft.Icons.INSIGHTS, label="Métricas"),
        ],
        selected_index=0,
        bgcolor="#161b22",
        indicator_color=ft.Colors.BLUE_900,
        on_change=on_nav_change,
    )

    page.add(
        ft.Column(
            [header, content_area],
            expand=True,
            spacing=0,
        )
    )
    page.navigation_bar = nav_bar
    page.update()


if __name__ == '__main__':
    ft.app(target=main)