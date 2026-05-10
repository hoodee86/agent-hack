"linux_agent package"

from __future__ import annotations

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
	"ignore",
	message=(
		r"The default value of `allowed_objects` will change in a future "
		r"version\..*"
	),
	category=LangChainPendingDeprecationWarning,
)
