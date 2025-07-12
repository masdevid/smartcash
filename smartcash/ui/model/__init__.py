"""
File: smartcash/ui/model/__init__.py
Deskripsi: Model UI module exports
"""

from .backbone import (
    BackboneUIModule,
    create_backbone_uimodule,
    get_backbone_uimodule,
    create_backbone_ui
)

__all__ = [
    # Backbone UIModule
    'BackboneUIModule',
    'create_backbone_uimodule',
    'get_backbone_uimodule',
    'create_backbone_ui'
]