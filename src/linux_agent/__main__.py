"""Allow ``python -m linux_agent`` to invoke the CLI."""
from linux_agent.app import main
import sys

sys.exit(main())
