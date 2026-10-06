"""Gráficos do post a partir de results/runs.csv, em versão clara e escura.

    uv run python -m eval.plot                    # camada 1: recuperação
    uv run python -m eval.plot --metric answer    # camada 2: resposta do modelo

Gera <métrica>_geral_*.png (uma linha por versão) e <métrica>_por_cenario_*.png
(small multiples, um painel por cenário). Paleta categórica validada para
daltonismo; aqua e amarelo têm contraste baixo no claro, por isso o post traz a
tabela junto.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scenarios.library import core_messages  # noqa: E402
from scenarios.model import WEEKS  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
CORE_MESSAGES = len(core_messages())

BASELINE = ("v0_no_memory", "0. Sem memória (referência)", "X")
VERSIONS = {  # ordem fixa: a cor segue a versão, nunca a posição no ranking
    "v1_naive": ("1. Ingênua", "o"),
    "v2_keyed": ("2. Chave de sujeito", "s"),
    "v3_decay_rank": ("3. Decaimento no ranking", "^"),
    "v4_decay_tier": ("4. Decaimento no tier", "D"),
}
SCENARIOS = {
    "update": "Mudança de valor",
    "scope": "Escopo",
    "rare_fact": "Regra dita uma vez",
    "injection": "Injeção externa",
}
THEMES = {
    "claro": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "escuro": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}


METRICS = {
    "retrieval": ("retrieved_ok", "A memória certa chegou ao contexto?",
                  "Memória certa no contexto, por cenário"),
    "answer": ("passed", "O modelo respondeu certo?", "Resposta certa, por cenário"),
}


def load(path: Path, column: str) -> tuple[dict, int]:
    """acc[(scenario|None, version, noise, gap_days)] = taxa média entre seeds; e o nº de seeds."""
    buckets: dict[tuple, list[bool]] = defaultdict(list)
    seeds = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            if r[column] == "":
                continue
            ok = r[column] == "True"
            noise, scale = int(r["noise_per_week"]), float(r.get("gap_days") or 0)
            seeds.add(r["seed"])
            buckets[(None, r["version"], noise, scale)].append(ok)
            buckets[(r["scenario"], r["version"], noise, scale)].append(ok)
    return {k: 100 * mean(v) for k, v in buckets.items()}, len(seeds)


def at_scale(acc: dict, scale: float) -> dict:
    """Fatia de uma escala de tempo, com a chave antiga (scenario, version, noise)."""
    return {k[:3]: v for k, v in acc.items() if k[3] == scale}


def at_noise(acc: dict, noise: int) -> dict:
    """Fatia de um nível de ruído; o eixo x passa a ser a escala de tempo."""
    return {(k[0], k[1], k[3]): v for k, v in acc.items() if k[2] == noise}


def _style(ax, theme: dict, noises: list, labels: list[str] | None = None) -> None:
    ax.set_facecolor(theme["surface"])
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]])
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels(labels or [str(CORE_MESSAGES + n * WEEKS) for n in noises])
    ax.grid(axis="y", color=theme["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=10)


def _lines(ax, acc: dict, scenario, noises: list[int], theme: dict) -> None:
    version, _, marker = BASELINE
    if any((scenario, version, n) in acc for n in noises):  # referência em cinza, atrás das versões
        ax.plot(range(len(noises)), [acc.get((scenario, version, n)) for n in noises],
                color=theme["muted"], linewidth=2, marker=marker, markersize=7, alpha=0.7,
                markeredgecolor=theme["surface"], markeredgewidth=1, zorder=1)
    # deslocamento horizontal pequeno por versão: empates (ex.: todas em 100%) continuam visíveis
    for i, ((version, (_, marker)), color) in enumerate(zip(VERSIONS.items(), theme["series"])):
        ys = [acc.get((scenario, version, n)) for n in noises]
        xs = [x + (i - 1.5) * 0.07 for x in range(len(noises))]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=theme["surface"], markeredgewidth=2,  # anel de superfície
                solid_joinstyle="round", solid_capstyle="round")


def _legend(fig, theme: dict, with_baseline: bool, y: float = 0.905) -> None:
    handles = [
        plt.Line2D([], [], color=c, linewidth=2, marker=m, markersize=7,
                   markeredgecolor=theme["surface"], markeredgewidth=2)
        for (_, m), c in zip(VERSIONS.values(), theme["series"])
    ]
    labels = [label for label, _ in VERSIONS.values()]
    if with_baseline:
        handles.append(plt.Line2D([], [], color=theme["muted"], linewidth=2, marker=BASELINE[2],
                                  markersize=7, alpha=0.7))
        labels.append(BASELINE[1])
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=3 if with_baseline else 4, frameon=False, fontsize=10,
                     handlelength=2.2, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(theme["text"])


def plot_overall(acc: dict, seeds: int, noises: list[int], theme_name: str, out: Path, title: str,
                 subtitle: str | None = None) -> None:
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.08, right=0.97)
    _style(ax, theme, noises)
    _lines(ax, acc, None, noises, theme)
    ax.set_xlabel("Mensagens no histórico do agente", color=theme["muted"], fontsize=10, labelpad=10)
    fig.text(0.08, 0.95, title,
             color=theme["text"], fontsize=15, fontweight="bold")
    fig.text(0.08, 0.915, subtitle or f"Média de {seeds} execuções, 28 perguntas cada, conforme o histórico cresce; 5 memórias por pergunta",
             color=theme["muted"], fontsize=10)
    _legend(fig, theme, any(k[1] == BASELINE[0] for k in acc))
    fig.savefig(out, facecolor=theme["surface"])
    plt.close(fig)


def plot_by_scenario(acc: dict, seeds: int, noises: list[int], theme_name: str, out: Path,
                     title: str) -> None:
    theme = THEMES[theme_name]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor(theme["surface"])
    fig.subplots_adjust(top=0.76, bottom=0.1, left=0.08, right=0.97, hspace=0.32, wspace=0.12)
    for ax, (scenario, panel_title) in zip(axes.flat, SCENARIOS.items()):
        _style(ax, theme, noises)
        _lines(ax, acc, scenario, noises, theme)
        ax.set_title(panel_title, color=theme["text"], fontsize=12, loc="left", pad=8)
    fig.supxlabel("Mensagens no histórico do agente", color=theme["muted"], fontsize=10, y=0.02)
    fig.text(0.08, 0.955, title, color=theme["text"], fontsize=15, fontweight="bold")
    fig.text(0.08, 0.92, f"Média de {seeds} execuções; 5 memórias recuperadas por pergunta",
             color=theme["muted"], fontsize=10)
    _legend(fig, theme, any(k[1] == BASELINE[0] for k in acc))
    fig.savefig(out, facecolor=theme["surface"])
    plt.close(fig)


def _span(gap: float) -> str:
    """Idade da regra na última pergunta: 7 semanas de conversa + a pausa."""
    days = 49 + gap
    return f"{days / 7:.0f} semanas" if days < 120 else f"{days / 30.4:.0f} meses"


def plot_time(acc: dict, seeds: int, scenario: str, noises: list[int], theme_name: str, out: Path,
              title: str) -> None:
    """Small multiples por tamanho de histórico; eixo x = tempo coberto pela conversa."""
    theme = THEMES[theme_name]
    scales = sorted({k[3] for k in acc})
    fig, axes = plt.subplots(1, len(noises), figsize=(4 * len(noises) + 1, 5), dpi=200, sharey=True)
    fig.patch.set_facecolor(theme["surface"])
    fig.subplots_adjust(top=0.66, bottom=0.17, left=0.07, right=0.98, wspace=0.12)
    for ax, noise in zip(axes, noises):
        _style(ax, theme, scales, [_span(sc) for sc in scales])
        _lines(ax, at_noise(acc, noise), scenario, scales, theme)
        ax.set_title(f"{CORE_MESSAGES + noise * WEEKS} mensagens", color=theme["text"], fontsize=12,
                     loc="left", pad=8)
    fig.supxlabel("Idade da regra na última pergunta (atividade recente igual em todos)",
                  color=theme["muted"], fontsize=10, y=0.03)
    fig.text(0.07, 0.94, title, color=theme["text"], fontsize=15, fontweight="bold")
    fig.text(0.07, 0.895, f"Média de {seeds} execuções; 5 memórias recuperadas por pergunta",
             color=theme["muted"], fontsize=10)
    _legend(fig, theme, any(k[1] == BASELINE[0] for k in acc), y=0.86)
    fig.savefig(out, facecolor=theme["surface"])
    plt.close(fig)


def load_stale_share(path: Path) -> dict:
    """acc[(None, version, noise)] = % médio do contexto com valor obsoleto, sem pausa."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["stale_share"] and float(r.get("gap_days") or 0) == 0:
                buckets[(None, r["version"], int(r["noise_per_week"]))].append(float(r["stale_share"]))
    return {k: 100 * mean(v) for k, v in buckets.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRICS), default="retrieval")
    args = ap.parse_args()
    column, title, title_by_scenario = METRICS[args.metric]
    full, seeds = load(RESULTS / "runs.csv", column)
    acc = at_scale(full, 0.0)
    noises = sorted({k[2] for k in acc})
    scales = sorted({k[3] for k in full})
    for theme in THEMES:
        if len(scales) > 1:
            plot_time(full, seeds, "rare_fact", [n for n in (10, 50, 100) if n in noises], theme,
                      RESULTS / f"{args.metric}_regra_rara_tempo_{theme}.png",
                      "Regra dita uma vez, conforme ela envelhece")
        plot_overall(acc, seeds, noises, theme, RESULTS / f"{args.metric}_geral_{theme}.png", title)
        if args.metric == "retrieval":
            stale = load_stale_share(RESULTS / "runs.csv")
            plot_overall(stale, seeds, sorted({k[2] for k in stale}), theme,
                         RESULTS / f"valor_obsoleto_{theme}.png",
                         "Quanto do contexto ainda fala a região antiga",
                         f"Depois da migração; quanto menor, melhor. Média de {seeds} execuções; 5 memórias por pergunta")
        plot_by_scenario(acc, seeds, noises, theme, RESULTS / f"{args.metric}_por_cenario_{theme}.png",
                         title_by_scenario)
    print(f"gráficos salvos em {RESULTS}")


if __name__ == "__main__":
    main()
