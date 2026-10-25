"""Explicit Earth Engine initialization and bounded request retry."""

import logging
import time

LOG = logging.getLogger(__name__)


def retry(function, attempts=5, *, retry_if=None):
    for attempt in range(attempts):
        try:
            return function()
        except Exception as error:
            if attempt == attempts - 1 or (retry_if is not None and not retry_if(error)):
                raise
            LOG.warning("GEE request failed; retry %d/%d", attempt + 1, attempts)
            time.sleep(min(2**attempt, 16))


def initialize(project, endpoint=None):
    import ee

    kwargs = {"project": project}
    if endpoint:
        kwargs["opt_url"] = endpoint
    ee.Initialize(**kwargs)
    ee.data.setDeadline(180000)
    return ee
