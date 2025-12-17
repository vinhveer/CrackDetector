from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.pipeline.models import BoxResult


@dataclass
class PromptConfig:
    primary: str = "hairline crack on {material} surface"
    secondary: List[str] = field(
        default_factory=lambda: ["tiny surface fracture on {material}", "narrow linear crack", "branching crack pattern"],
    )
    adaptive: bool = True
    min_detections_for_keep: int = 2


def _default_damage_type_prompts() -> Dict[str, List[str]]:
    return {
        "crack": [
            "crack on {material} surface",
            "hairline crack on {material} surface",
            "narrow linear crack on {material} surface",
        ],
        "spalling": [
            "spalling on {material} surface",
            "chipped surface on {material}",
            "surface spalling",
        ],
        "peeling": [
            "peeling on {material} surface",
            "peeling paint on {material} surface",
            "paint peeling",
        ],
        "surface deterioration": [
            "surface deterioration on {material} surface",
            "surface degradation on {material} surface",
            "eroded {material} surface",
        ],
        "stain": [
            "stain on {material} surface",
            "water stain on {material} surface",
            "dirty stain",
        ],
    }


@dataclass
class DamagePromptConfig:
    enabled: bool = False
    mode: str = "one_pass"
    one_pass: str = "surface damage on {material} surface"
    types: List[str] = field(
        default_factory=lambda: ["crack", "spalling", "peeling", "surface deterioration", "stain"],
    )
    type_prompts: Dict[str, List[str]] = field(default_factory=_default_damage_type_prompts)


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

            damage_cfg = config.get("damage")
            if isinstance(damage_cfg, dict):
                types_raw = damage_cfg.get("types")
                types = types_raw if isinstance(types_raw, list) else DamagePromptConfig().types
                type_prompts_raw = damage_cfg.get("type_prompts")
                type_prompts = type_prompts_raw if isinstance(type_prompts_raw, dict) else DamagePromptConfig().type_prompts
                self.damage = DamagePromptConfig(
                    enabled=bool(damage_cfg.get("enabled", False)),
                    mode=str(damage_cfg.get("mode", DamagePromptConfig.mode)),
                    one_pass=str(damage_cfg.get("one_pass", DamagePromptConfig.one_pass)),
                    types=types,
                    type_prompts=type_prompts,
                )
            else:
                self.damage = DamagePromptConfig()
        else:
            self.config = config or PromptConfig()
            self.damage = DamagePromptConfig()

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

    def damage_prompt_map(self, material_type: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, List[str]]:
        mat = material_type or self.material_type
        selected_mode = (mode or self.damage.mode or "one_pass").strip().lower()

        if selected_mode in {"one", "one_pass", "single", "generic"}:
            return {"surface_damage": [self.damage.one_pass.format(material=mat)]}

        out: Dict[str, List[str]] = {}
        for damage_type in self.damage.types:
            prompts = self.damage.type_prompts.get(damage_type)
            if not prompts:
                prompts = [f"{damage_type} on {{material}} surface"]
            out[damage_type] = [p.format(material=mat) for p in prompts]
        return out

    def detect_damage(
        self,
        detector: Any,
        image,
        material_type: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, List[BoxResult]]:
        prompt_map = self.damage_prompt_map(material_type=material_type, mode=mode)
        out: Dict[str, List[BoxResult]] = {}
        for name, prompts in prompt_map.items():
            out[name] = detector.detect(image, prompts)
        return out

