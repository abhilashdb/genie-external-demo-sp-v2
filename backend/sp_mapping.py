"""Map an sp_label to the service principal's credentials."""

from __future__ import annotations

from typing import Tuple

from .config import settings


def resolve_sp(sp_label: str) -> Tuple[str, str, str, str]:
    """Return (client_id, client_secret, dealership_name, sp_app_id).

    Raises KeyError if the label is unknown.
    """
    label = (sp_label or "").lower().strip()
    if label == "northstar":
        return (
            settings.sp_northstar_client_id,
            settings.sp_northstar_secret,
            settings.sp_northstar_dealership,
            settings.sp_northstar_app_id,
        )
    if label == "sunrise":
        return (
            settings.sp_sunrise_client_id,
            settings.sp_sunrise_secret,
            settings.sp_sunrise_dealership,
            settings.sp_sunrise_app_id,
        )
    raise KeyError(f"Unknown sp_label: {sp_label!r}")
