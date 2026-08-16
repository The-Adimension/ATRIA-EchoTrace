"""CLI argument handling.

Covers the parser surface and the no-argument default, which is the click-and-run
path the launchers rely on. Subcommands that start a server, train, or upload are
verified through their own modules and by running them directly, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atria_echotrace.cli import build_parser, main


def test_no_arguments_defaults_to_serve() -> None:
    """`atria` with no arguments must serve — the launchers depend on it."""
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    # The default is applied in main() before parsing; assert the substitution itself.
    import atria_echotrace.cli as cli_module

    captured: dict[str, object] = {}

    def fake_serve(parsed, settings):
        captured["command"] = parsed.command
        return 0

    original = cli_module.cmd_serve
    cli_module.cmd_serve = fake_serve
    try:
        # Rebuild so the parser binds the replacement.
        assert main([]) == 0
    finally:
        cli_module.cmd_serve = original
    assert captured["command"] == "serve"


def test_every_documented_subcommand_parses() -> None:
    parser = build_parser()
    for argv in (
        ["serve"],
        ["doctor"],
        ["ingest", "camus", "--source", "a", "--output", "b"],
        ["ingest", "echonet", "--source", "a", "--output", "b"],
        ["train", "--output-dir", "out"],
        ["evaluate"],
        ["publish-adapter", "--folder", "f", "--repo", "o/n"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_doctor_reports_the_real_dataset(capsys, sample_dataset_dir: Path) -> None:
    assert main(["--dataset-dir", str(sample_dataset_dir), "doctor"]) == 0
    output = capsys.readouterr().out
    assert "frames        : 50" in output
    assert "cases         : 25" in output
    # The integrity warning must be visible, not buried.
    assert "transposed ED/ES labels" in output
    assert "PASS" in output


def test_doctor_fails_on_a_missing_dataset(capsys, tmp_path: Path) -> None:
    assert main(["--dataset-dir", str(tmp_path), "doctor"]) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize(("given", "expected"), [("all", None), ("test", "test")])
def test_evaluate_split_all_is_translated_to_none(given: str, expected: str | None) -> None:
    """`--split all` means "every frame", which the frame selector expresses as None."""
    import atria_echotrace.cli as cli_module

    captured: dict[str, object] = {}

    def fake_evaluate(parsed, settings):
        captured["split"] = parsed.split
        return 0

    original = cli_module.cmd_evaluate
    cli_module.cmd_evaluate = fake_evaluate
    try:
        assert main(["evaluate", "--split", given]) == 0
    finally:
        cli_module.cmd_evaluate = original
    assert captured["split"] == expected


def test_ingest_defaults_reproduce_the_training_corpus() -> None:
    """Defaults must match what the reference scripts were run with."""
    args = build_parser().parse_args(["ingest", "camus", "--source", "a", "--output", "b"])
    assert args.points == 30  # 30-point polygons, as in the corpus
    assert args.target_size == 224  # EchoNet 112 -> 224 Lanczos, as in the corpus


def test_ingest_accepts_the_unified_merge_step() -> None:
    """The merge step is what actually produced the 22k-frame training corpus."""
    args = build_parser().parse_args(
        [
            "ingest",
            "unified",
            "--camus-processed",
            "c",
            "--echonet-processed",
            "e",
            "--output",
            "u",
        ]
    )
    assert args.dataset == "unified"
    assert args.camus_processed == "c"
    assert args.echonet_processed == "e"


def test_ingest_requires_source_for_raw_datasets(capsys) -> None:
    """camus/echonet need --source; unified does not."""
    assert main(["ingest", "camus", "--output", "out"]) == 1
    assert "needs --source" in capsys.readouterr().err


def test_train_defaults_match_the_notebook() -> None:
    args = build_parser().parse_args(["train", "--output-dir", "out"])
    assert args.epochs == 10
    assert args.learning_rate == pytest.approx(2e-4)
    assert args.train_samples == 1000
    assert args.val_samples == 200
    assert args.structure == "LV"


def test_publish_defaults_to_gated_and_unconfirmed() -> None:
    args = build_parser().parse_args(["publish-adapter", "--folder", "f", "--repo", "o/n"])
    assert args.public is False  # gated by default
    assert args.yes is False  # never uploads without explicit confirmation


def test_unknown_command_exits() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["not-a-command"])
