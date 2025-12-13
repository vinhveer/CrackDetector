from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptConfig:
    primary: str = "hairline crack on {material} surface"
    secondary: List[str] = field(
        default_factory=lambda: ["tiny surface fracture on {material}", "narrow linear crack", "branching crack pattern"],
    )
    adaptive: bool = True
    min_detections_for_keep: int = 2


class PromptManager:
    """Generates and adapts textual prompts for GroundingDINO."""

    def __init__(self, config: Optional[PromptConfig | Dict] = None, material_type: str = "concrete") -> None:
        self.material_type = material_type
        if isinstance(config, dict):
            self.config = PromptConfig(
                primary=config.get("primary", PromptConfig.primary),
                secondary=config.get("secondary", PromptConfig().secondary),
                adaptive=config.get("adaptive", True),
                min_detections_for_keep=config.get("min_detections_for_keep", 2),
            )
        else:
            self.config = config or PromptConfig()

    def generate(self, material_type: Optional[str] = None) -> List[str]:
        mat = material_type or self.material_type
        prompts: List[str] = [self.config.primary.format(material=mat)]
        prompts.extend([p.format(material=mat) for p in self.config.secondary])
        return prompts

    def adapt(self, material_type: str, detections: int, noise_level: float = 0.0) -> List[str]:
        if not self.config.adaptive:
            return self.generate(material_type)
        if detections >= self.config.min_detections_for_keep and noise_level < 0.5:
            return self.generate(material_type)
        adapted: List[str] = [
            "fine hairline crack on {material}",
            "micro fracture line on {material} surface",
            "subtle narrow crack with low contrast",
        ]
        if noise_level > 0.5:
            adapted.append("clean linear crack with minimal background noise on {material}")
        return [p.format(material=material_type) for p in adapted]

    def multi_prompt_set(self, material_type: str, detections: int = 0, noise_level: float = 0.0) -> List[str]:
        primary = self.generate(material_type)
        secondary = self.adapt(material_type, detections, noise_level)
        merged = primary + [p for p in secondary if p not in primary]
        return merged

