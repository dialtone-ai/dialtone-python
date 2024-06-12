"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Dialtone."""


if __name__ == "__main__":
    main(prog_name="dialtone_python")  # pragma: no cover
