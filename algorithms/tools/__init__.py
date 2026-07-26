"""
Cardiomni Agent Tool Suite

Reusable specialist tools for the Cardiomni agent's four-stage SOP pipeline.
These are NOT baselines; they are tools the agent calls during Stages 2 and 4.

Per PROPOSAL.md §2.6 and §5, specialist models are shared tools available to
every harness/agent. The Cardiomni agent's orchestration of these tools is what
differentiates it from pure VLM reasoning.

Available tools:
- vessel_segmentation: 2D XCA binary vessel/background segmentation (CM-UNet)
- diameter_qca: QCA-style minimum lumen diameter + %DS quantification
- stenosis_detection: DeepCORO-CLIP stenosis detector (weights unavailable)
"""

__all__ = ["segment_vessels", "quantify_stenosis", "detect_stenosis"]

from .vessel_segmentation import segment_vessels
from .diameter_qca import quantify_stenosis
from .stenosis_detection import detect_stenosis
