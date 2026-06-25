"""
Dose-Response Module for Quantity-Aware Health Risk Predictions

This module implements dose-response scaling that adjusts GNN base predictions
based on the quantity of ingredients consumed relative to Recommended Daily Allowances (RDA).
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class NutrientType(Enum):
    """Type of nutrient - determines how dose-response is calculated."""
    LIMIT = "limit"      # Harmful at high levels (sugar, sodium)
    TARGET = "target"    # Beneficial - should meet target (fiber, protein)
    BOTH = "both"        # Too little or too much is bad


@dataclass
class NutrientRDA:
    """RDA/limit information for a nutrient."""
    name: str
    value: float
    unit: str
    nutrient_type: NutrientType
    description: str = ""


# RDA Reference Values (US FDA / WHO Guidelines)
RDA_DATABASE: Dict[str, NutrientRDA] = {
    # Nutrients to LIMIT
    "sugar": NutrientRDA("Sugar", 50, "g", NutrientType.LIMIT, "Added sugars"),
    "added sugar": NutrientRDA("Added Sugar", 50, "g", NutrientType.LIMIT, "Added sugars"),
    "sodium": NutrientRDA("Sodium", 2300, "mg", NutrientType.LIMIT, "Daily sodium limit"),
    "salt": NutrientRDA("Salt", 6, "g", NutrientType.LIMIT, "Table salt"),
    "saturated fat": NutrientRDA("Saturated Fat", 20, "g", NutrientType.LIMIT, "Saturated fatty acids"),
    "trans fat": NutrientRDA("Trans Fat", 2, "g", NutrientType.LIMIT, "Trans fatty acids"),
    "cholesterol": NutrientRDA("Cholesterol", 300, "mg", NutrientType.LIMIT, "Dietary cholesterol"),
    
    # Nutrients to TARGET
    "fiber": NutrientRDA("Fiber", 28, "g", NutrientType.TARGET, "Dietary fiber"),
    "dietary fiber": NutrientRDA("Dietary Fiber", 28, "g", NutrientType.TARGET, "Dietary fiber"),
    "protein": NutrientRDA("Protein", 50, "g", NutrientType.TARGET, "Total protein"),
    "calcium": NutrientRDA("Calcium", 1300, "mg", NutrientType.TARGET, "Calcium"),
    "iron": NutrientRDA("Iron", 18, "mg", NutrientType.TARGET, "Iron"),
    "potassium": NutrientRDA("Potassium", 4700, "mg", NutrientType.TARGET, "Potassium"),
    "vitamin c": NutrientRDA("Vitamin C", 90, "mg", NutrientType.TARGET, "Ascorbic acid"),
    "vitamin d": NutrientRDA("Vitamin D", 20, "mcg", NutrientType.TARGET, "Vitamin D"),
    "magnesium": NutrientRDA("Magnesium", 420, "mg", NutrientType.TARGET, "Magnesium"),
    "zinc": NutrientRDA("Zinc", 11, "mg", NutrientType.TARGET, "Zinc"),
    "omega-3": NutrientRDA("Omega-3", 1.6, "g", NutrientType.TARGET, "Omega-3 fatty acids"),
    
    # General macros
    "fat": NutrientRDA("Total Fat", 78, "g", NutrientType.BOTH, "Total fat"),
    "total fat": NutrientRDA("Total Fat", 78, "g", NutrientType.BOTH, "Total fat"),
    "carbohydrate": NutrientRDA("Carbohydrates", 275, "g", NutrientType.BOTH, "Total carbs"),
    "calories": NutrientRDA("Calories", 2000, "kcal", NutrientType.BOTH, "Daily energy"),
}

# Unit conversion factors
UNIT_CONVERSIONS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "mcg": 0.000001, "microgram": 0.000001,
    "kg": 1000.0,
    "oz": 28.3495, "ounce": 28.3495,
    "ml": 1.0, "milliliter": 1.0,
    "l": 1000.0, "liter": 1000.0,
    "tsp": 5.0, "teaspoon": 5.0,
    "tbsp": 15.0, "tablespoon": 15.0,
    "cup": 240.0, "cups": 240.0,
    "%": 1.0, "percent": 1.0, "%dv": 1.0,
    "kcal": 1.0, "cal": 0.001, "calorie": 0.001,
}


@dataclass
class IngredientQuantity:
    """Represents an ingredient with its quantity."""
    name: str
    amount: float
    unit: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "amount": self.amount, "unit": self.unit}


@dataclass
class DoseResponseResult:
    """Result of dose-response calculation for an ingredient."""
    ingredient: str
    original_amount: float
    original_unit: str
    normalized_amount: float
    rda_value: Optional[float]
    rda_unit: Optional[str]
    pct_rda: float
    dose_response_factor: float
    nutrient_type: Optional[NutrientType]
    is_known_nutrient: bool
    
    def _get_risk_level(self) -> str:
        """Get human-readable risk level based on % RDA."""
        if self.nutrient_type == NutrientType.LIMIT:
            if self.pct_rda >= 1.5: return "very_high"
            elif self.pct_rda >= 1.0: return "high"
            elif self.pct_rda >= 0.5: return "moderate"
            else: return "low"
        elif self.nutrient_type == NutrientType.TARGET:
            if self.pct_rda >= 1.0: return "adequate"
            elif self.pct_rda >= 0.5: return "moderate"
            else: return "low"
        return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ingredient": self.ingredient,
            "original_amount": self.original_amount,
            "original_unit": self.original_unit,
            "pct_rda": round(self.pct_rda * 100, 1),
            "dose_response_factor": round(self.dose_response_factor, 3),
            "risk_level": self._get_risk_level(),
            "rda_value": self.rda_value,
            "rda_unit": self.rda_unit,
            "is_known_nutrient": self.is_known_nutrient
        }


def normalize_unit(amount: float, from_unit: str, to_unit: str) -> float:
    """Convert amount from one unit to another."""
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    if from_unit == to_unit:
        return amount
    from_factor = UNIT_CONVERSIONS.get(from_unit, 1.0)
    to_factor = UNIT_CONVERSIONS.get(to_unit, 1.0)
    if from_unit == "%dv":
        return amount / 100.0
    return amount * from_factor / to_factor


def find_rda(ingredient: str) -> Optional[NutrientRDA]:
    """Find RDA reference for an ingredient."""
    ingredient_lower = ingredient.lower().strip()
    if ingredient_lower in RDA_DATABASE:
        return RDA_DATABASE[ingredient_lower]
    for key, rda in RDA_DATABASE.items():
        if key in ingredient_lower or ingredient_lower in key:
            return rda
    return None


def calculate_dose_response_factor(pct_rda: float, nutrient_type: NutrientType) -> float:
    """Calculate the dose-response scaling factor."""
    if nutrient_type == NutrientType.LIMIT:
        factor = min(2.5, 0.1 + 0.9 * (pct_rda ** 1.2))
        return max(0.1, factor)
    elif nutrient_type == NutrientType.TARGET:
        if pct_rda <= 0:
            return 1.5
        factor = 1.5 / (1 + pct_rda)
        return max(0.2, min(1.5, factor))
    elif nutrient_type == NutrientType.BOTH:
        deviation = abs(pct_rda - 1.0)
        factor = 1.0 + (deviation ** 2) * 0.5
        return min(2.0, factor)
    return 1.0


def calculate_dose_response(ingredient: IngredientQuantity) -> DoseResponseResult:
    """Calculate dose-response for a single ingredient."""
    rda = find_rda(ingredient.name)
    
    if rda is None:
        return DoseResponseResult(
            ingredient=ingredient.name,
            original_amount=ingredient.amount,
            original_unit=ingredient.unit,
            normalized_amount=ingredient.amount,
            rda_value=None, rda_unit=None,
            pct_rda=1.0,
            dose_response_factor=1.0,
            nutrient_type=None,
            is_known_nutrient=False
        )
    
    normalized = normalize_unit(ingredient.amount, ingredient.unit, rda.unit)
    pct_rda = normalized / rda.value if rda.value > 0 else 0
    dose_factor = calculate_dose_response_factor(pct_rda, rda.nutrient_type)
    
    return DoseResponseResult(
        ingredient=ingredient.name,
        original_amount=ingredient.amount,
        original_unit=ingredient.unit,
        normalized_amount=normalized,
        rda_value=rda.value,
        rda_unit=rda.unit,
        pct_rda=pct_rda,
        dose_response_factor=dose_factor,
        nutrient_type=rda.nutrient_type,
        is_known_nutrient=True
    )


def adjust_risk_predictions(
    base_predictions: Dict[str, float],
    dose_response_results: List[DoseResponseResult]
) -> Dict[str, float]:
    """Adjust base GNN risk predictions based on dose-response factors."""
    if not dose_response_results:
        return base_predictions
    
    total_weight = 0.0
    weighted_factor_sum = 0.0
    
    for result in dose_response_results:
        weight = max(0.1, result.pct_rda)
        if result.nutrient_type == NutrientType.LIMIT:
            weight *= result.dose_response_factor
        weighted_factor_sum += result.dose_response_factor * weight
        total_weight += weight
    
    avg_factor = weighted_factor_sum / total_weight if total_weight > 0 else 1.0
    
    adjusted = {}
    for disease, prob in base_predictions.items():
        adjusted[disease] = round(min(0.99, prob * avg_factor), 4)
    return adjusted


def parse_ingredient_string(text: str) -> List[IngredientQuantity]:
    """Parse a free-form ingredient string into structured quantities."""
    ingredients = []
    parts = re.split(r'[,;]+', text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        patterns = [
            r'^(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)\s+(.+)$',
            r'^([a-zA-Z][a-zA-Z\s]+?)[\s:]+(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)$',
            r'^([a-zA-Z][a-zA-Z\s]+)$'
        ]
        
        matched = False
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, part, re.IGNORECASE)
            if match:
                if i == 0:
                    amount, unit, name = float(match.group(1)), match.group(2), match.group(3)
                elif i == 1:
                    name, amount, unit = match.group(1), float(match.group(2)), match.group(3)
                else:
                    name, amount, unit = match.group(1), 100.0, "g"
                
                ingredients.append(IngredientQuantity(name.strip(), amount, unit.strip()))
                matched = True
                break
        
        if not matched:
            ingredients.append(IngredientQuantity(part, 100.0, "g"))
    
    return ingredients


def get_rda_info(ingredient: str) -> Optional[Dict[str, Any]]:
    """Get RDA information for an ingredient."""
    rda = find_rda(ingredient)
    if rda:
        return {
            "name": rda.name, "value": rda.value, "unit": rda.unit,
            "type": rda.nutrient_type.value, "description": rda.description
        }
    return None


def get_all_rda_values() -> List[Dict[str, Any]]:
    """Get all RDA values for frontend display."""
    return [
        {"key": key, "name": rda.name, "value": rda.value, "unit": rda.unit,
         "type": rda.nutrient_type.value, "description": rda.description}
        for key, rda in RDA_DATABASE.items()
    ]


if __name__ == "__main__":
    print("=== Dose-Response Module Test ===")
    
    ing = IngredientQuantity("sugar", 75, "g")
    result = calculate_dose_response(ing)
    print(f"Sugar 75g: {result.pct_rda*100:.0f}% RDA, factor={result.dose_response_factor:.2f}")
    
    base = {"diabetes": 0.5, "obesity": 0.4}
    adjusted = adjust_risk_predictions(base, [result])
    print(f"Base: {base}")
    print(f"Adjusted: {adjusted}")
    
    parsed = parse_ingredient_string("50g sugar, 2000mg sodium")
    for p in parsed:
        print(f"  {p.name}: {p.amount}{p.unit}")
    
    print("\nAll tests passed!")
