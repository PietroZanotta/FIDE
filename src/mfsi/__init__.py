"""MFSI reusable core.

Keep package import deliberately light.  Experiment code imports concrete
components from their modules (``mfsi.design``, ``mfsi.projection``, ...), so
``import mfsi`` must not depend on optional convenience re-exports.
"""

import jax

jax.config.update("jax_enable_x64", True)