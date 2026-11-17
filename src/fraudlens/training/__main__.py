"""Allow ``python -m fraudlens.training <step>``."""

import sys

from fraudlens.training.cli import main

sys.exit(main())
