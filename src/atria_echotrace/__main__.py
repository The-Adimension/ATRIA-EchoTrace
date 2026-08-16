"""Allow ``python -m atria_echotrace`` as an alias for the ``atria`` command.

The launchers use this form so they work before (or without) the console script
being on PATH.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
