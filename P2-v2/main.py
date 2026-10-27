"""Orquestador del proyecto de detección de fraude.

Ejecuta la matriz de 8 experimentos:
1. Carga y auditoría del dataset.
2. División estratificada 70/30 (test congelado).
3. Para cada experimento: búsqueda de hiperparámetros con CV=5 guiada por
   recall, reentrenamiento sobre el 70% y evaluación sobre el 30%.
4. Exportación de tablas, métricas y gráficos en ``Graficos/``.

Ejemplos de uso:
    python main.py                      # dataset completo (puede tardar horas)
    python main.py --quick              # submuestra + rejillas reducidas
    python main.py --sample-size 20000  # submuestra personalizada
    python main.py --experiments EXP_1,EXP_7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from Modulos import config
from Modulos.data_loader import load_data
from Modulos.data_splitter import split_data
from Modulos.evaluation.cross_validator import (
    cv_results_dataframe,
    run_search,
    summarize_search,
)
from Modulos.evaluation.metrics_calculator import calculate_metrics
from Modulos.evaluation.reporter import (
    plot_confusion_matrices,
    plot_cv_recall,
    plot_metric_comparison,
    save_cv_results,
    save_results,
)
from Modulos.experiments import build_experiments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparación de clasificadores para detección de fraude."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Submuestra estratificada para pruebas rápidas "
            f"(mínimo {config.MIN_SAMPLE_SIZE:,}). Por defecto usa todo el dataset."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Modo rápido: submuestra de {config.QUICK_SAMPLE_SIZE:,} registros "
            "y rejillas de hiperparámetros reducidas."
        ),
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Lista separada por comas de experimentos a ejecutar (p. ej. EXP_1,EXP_2).",
    )
    parser.add_argument(
        "--feature-score",
        choices=["mutual_info", "f_classif"],
        default=config.FEATURE_SCORE,
        help=(
            "Función de puntuación para SelectKBest. 'mutual_info' es la del "
            "SDD pero cuesta ~15-25 s por fold en el dataset completo; "
            "'f_classif' es mucho más rápida."
        ),
    )
    parser.add_argument(
        "--resampler",
        choices=["under", "smote"],
        default=config.DEFAULT_RESAMPLER,
        help="Técnica de compensación para los experimentos balanceados.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=config.N_JOBS,
        help="Número de procesos paralelos (-1 = todos los núcleos).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nivel de detalle de GridSearchCV/RandomizedSearchCV (0-3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directorio donde guardar tablas y gráficos.",
    )
    return parser.parse_args(argv)


def _print_header(title: str) -> None:
    print(), print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    sample_size = args.sample_size
    if sample_size is None:
        sample_size = (
            config.QUICK_SAMPLE_SIZE if args.quick else config.SAMPLE_SIZE
        )

    _print_header("DETECCIÓN DE FRAUDE - COMPARACIÓN DE CLASIFICADORES")
    print(f"Semilla global     : {config.SEED}")
    print(f"División train/test: {1 - config.TEST_SIZE:.0%}/{config.TEST_SIZE:.0%}")
    print(f"Validación cruzada : StratifiedKFold(n_splits={config.N_SPLITS})")
    print(f"Métrica principal  : Recall (clase fraude = 1)")
    print(f"Métrica desempate  : Precisión")
    print(f"Compensación       : {args.resampler}")
    print(f"Selección features : {args.feature_score}")
    print(f"Muestreo           : {'completo' if sample_size is None else f'{sample_size:,} registros'}")
    if sample_size is None and args.feature_score == "mutual_info":
        print(
            "\n[aviso] mutual_info sobre el dataset completo puede tardar horas. "
            "Considere --feature-score f_classif para una ejecución completa rápida."
        )
    print()

    # ------------------------------------------------------------------
    # 1. Datos
    # ------------------------------------------------------------------
    df = load_data(sample_size=sample_size, seed=config.SEED)
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=config.TEST_SIZE, seed=config.SEED
    )

    _print_header("PARTICIÓN CONGELADA")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(
        f"Fraudes en train: {int(y_train.sum()):,} "
        f"({y_train.mean():.4%})"
    )
    print(
        f"Fraudes en test : {int(y_test.sum()):,} "
        f"({y_test.mean():.4%})"
    )
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2. Experimentos
    # ------------------------------------------------------------------
    experiments = build_experiments(quick=args.quick, resampler=args.resampler)

    if args.experiments:
        requested = {
            item.strip().upper()
            for item in args.experiments.split(",")
            if item.strip()
        }
        experiments = [
            exp for exp in experiments if exp.exp_id.upper() in requested
        ]
        if not experiments:
            raise SystemExit(
                "No se seleccionó ningún experimento válido. "
                "Use, por ejemplo: --experiments EXP_1,EXP_2"
            )

    results: list[dict] = []
    cv_frames: list[pd.DataFrame] = []

    for experiment in experiments:
        _print_header(
            f"{experiment.exp_id} | {experiment.classifier} | "
            f"{experiment.compensation}"
        )
        pipeline = experiment.build_pipeline(feature_score=args.feature_score)
        print(f"Pipeline: {pipeline}")
        print(f"Candidatos a evaluar: {experiment.n_iter} x {config.N_SPLITS} folds")
        print()

        start = time.perf_counter()

        search = run_search(
            pipeline,
            experiment.param_distributions,
            X_train,
            y_train,
            n_iter=experiment.n_iter,
            n_splits=config.N_SPLITS,
            seed=config.SEED,
            n_jobs=args.jobs,
            verbose=args.verbose,
        )
        search_summary = summarize_search(search)

        # ``best_estimator_`` ya está reentrenado sobre todo X_train gracias
        # al refit de la búsqueda. El test permaneció congelado.
        y_pred = search.best_estimator_.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_pred, pos_label=config.POS_LABEL)

        elapsed = time.perf_counter() - start

        result = {
            "exp_id": experiment.exp_id,
            "classifier": experiment.classifier,
            "compensation": experiment.compensation,
            **search_summary,
            **test_metrics,
            "elapsed_s": float(elapsed),
        }
        results.append(result)
        cv_frames.append(cv_results_dataframe(search, experiment.exp_id))

        print()
        print(f"Mejores hiperparámetros: {json.dumps(search_summary['best_params'], default=str)}")
        print(
            f"CV recall : {search_summary['cv_recall_mean']:.4f} "
            f"± {search_summary['cv_recall_std']:.4f}"
        )
        print(
            f"CV precision: {search_summary['cv_precision_mean']:.4f} "
            f"± {search_summary['cv_precision_std']:.4f}"
        )
        print(
            f"TEST  recall={test_metrics['recall']:.4f} | "
            f"precision={test_metrics['precision']:.4f} | "
            f"specificity={test_metrics['specificity']:.4f} | "
            f"F1={test_metrics['f1']:.4f} | "
            f"FNR={test_metrics['fnr']:.4f}"
        )
        print(
            f"Matriz de confusión: TN={test_metrics['tn']:,} FP={test_metrics['fp']:,} "
            f"FN={test_metrics['fn']:,} TP={test_metrics['tp']:,}"
        )
        print(f"Tiempo: {elapsed:.1f} s")

    # ------------------------------------------------------------------
    # 3. Reportes
    # ------------------------------------------------------------------
    _print_header("GENERACIÓN DE REPORTES")
    output_dir = Path(args.output_dir)

    report = save_results(results, output_dir)
    cv_path = save_cv_results(cv_frames, output_dir)
    confusion_paths = plot_confusion_matrices(results, output_dir)
    metrics_path = plot_metric_comparison(results, output_dir)
    cv_plot_path = plot_cv_recall(results, output_dir)

    print(f"Tabla CSV        : {report['csv']}")
    print(f"Tabla Markdown   : {report['md']}")
    print(f"Resultados CV    : {cv_path}")
    for path in confusion_paths:
        print(f"Matriz confusión : {path}")
    print(f"Comparación      : {metrics_path}")
    print(f"Recall CV        : {cv_plot_path}")

    # ------------------------------------------------------------------
    # 4. Ranking final
    # ------------------------------------------------------------------
    _print_header("RANKING FINAL (criterio: recall, desempate: precisión)")
    ranking = report["dataframe"].sort_values(
        ["recall", "precision"], ascending=False
    )
    display_columns = [
        "exp_id",
        "classifier",
        "compensation",
        "recall",
        "precision",
        "specificity",
        "f1",
        "fnr",
    ]
    print(ranking[display_columns].to_string(index=False))
    print()
    best = ranking.iloc[0]
    print(
        f"Mejor configuración: {best['exp_id']} ({best['classifier']}, "
        f"{best['compensation']}) con recall={best['recall']:.4f} "
        f"y precisión={best['precision']:.4f}."
    )


if __name__ == "__main__":
    main()
