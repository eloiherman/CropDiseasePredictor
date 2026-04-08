import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, render_template, request, send_from_directory

from predict import get_feature_importance_map, get_model_class_count, make_prediction

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

BASE_DIR = Path(__file__).resolve().parent
DRAFTS_DIR = BASE_DIR / "code" / "drafts"


FEATURE_FIELDS = [
    {
        "name": "Elevation_Data",
        "label": "Elevation Data",
        "short_label": "Elevation",
        "placeholder": "55.0",
        "hint": "Approximate field elevation in the dataset. Typical range: 32.6 to 77.4.",
        "step": "any",
        "min": "0",
        "max": "100",
    },
    {
        "name": "Canopy_Coverage",
        "label": "Canopy Coverage",
        "short_label": "Canopy",
        "placeholder": "49.9",
        "hint": "Estimated plant coverage over the field surface. Typical range: 14.4 to 69.4.",
        "step": "any",
        "min": "0",
        "max": "100",
    },
    {
        "name": "NDVI",
        "label": "NDVI",
        "short_label": "NDVI",
        "placeholder": "0.50",
        "hint": "Vegetation index often used to measure crop vigor. Typical range: 0.40 to 0.60.",
        "step": "any",
        "min": "-1",
        "max": "1.5",
    },
    {
        "name": "SAVI",
        "label": "SAVI",
        "short_label": "SAVI",
        "placeholder": "0.40",
        "hint": "Soil-adjusted vegetation index that accounts for background soil brightness.",
        "step": "any",
        "min": "-1",
        "max": "1.5",
    },
    {
        "name": "Chlorophyll_Content",
        "label": "Chlorophyll Content",
        "short_label": "Chlorophyll",
        "placeholder": "1.00",
        "hint": "Relative chlorophyll reading linked to leaf health and photosynthesis.",
        "step": "any",
        "min": "0",
        "max": "8",
    },
    {
        "name": "Leaf_Area_Index",
        "label": "Leaf Area Index",
        "short_label": "Leaf Area",
        "placeholder": "1.71",
        "hint": "Leaf area relative to ground area. Typical range: 1.0 to 2.3.",
        "step": "any",
        "min": "0",
        "max": "6",
    },
    {
        "name": "Temperature",
        "label": "Temperature",
        "short_label": "Temperature",
        "placeholder": "25.0",
        "hint": "Air temperature near the crop canopy, in degrees Celsius.",
        "step": "any",
        "min": "0",
        "max": "50",
    },
    {
        "name": "Humidity",
        "label": "Humidity",
        "short_label": "Humidity",
        "placeholder": "60.0",
        "hint": "Relative humidity percentage recorded for field conditions.",
        "step": "any",
        "min": "0",
        "max": "100",
    },
    {
        "name": "Rainfall",
        "label": "Rainfall",
        "short_label": "Rainfall",
        "placeholder": "20.0",
        "hint": "Rainfall amount associated with the crop conditions.",
        "step": "any",
        "min": "0",
        "max": "300",
    },
    {
        "name": "Wind_Speed",
        "label": "Wind Speed",
        "short_label": "Wind Speed",
        "placeholder": "2.0",
        "hint": "Wind speed during the observation period.",
        "step": "any",
        "min": "0",
        "max": "20",
    },
    {
        "name": "Soil_Moisture",
        "label": "Soil Moisture",
        "short_label": "Soil Moisture",
        "placeholder": "20.0",
        "hint": "Moisture level in the soil profile. Typical range: 12.6 to 27.5.",
        "step": "any",
        "min": "0",
        "max": "40",
    },
    {
        "name": "Soil_pH",
        "label": "Soil pH",
        "short_label": "Soil pH",
        "placeholder": "6.5",
        "hint": "Acidity or alkalinity of the soil. Typical range: 6.16 to 6.84.",
        "step": "any",
        "min": "0",
        "max": "14",
    },
    {
        "name": "Organic_Matter",
        "label": "Organic Matter",
        "short_label": "Organic Matter",
        "placeholder": "2.0",
        "hint": "Estimated organic matter content in the soil.",
        "step": "any",
        "min": "0",
        "max": "30",
    },
    {
        "name": "Water_Flow",
        "label": "Water Flow",
        "short_label": "Water Flow",
        "placeholder": "25.0",
        "hint": "Water movement across or through the field area.",
        "step": "any",
        "min": "0",
        "max": "50",
    },
    {
        "name": "Weed_Coverage",
        "label": "Weed Coverage",
        "short_label": "Weed Coverage",
        "placeholder": "2.9",
        "hint": "Estimated weed presence competing with the wheat crop.",
        "step": "any",
        "min": "0",
        "max": "10",
    },
    {
        "name": "Crop_Stress_Indicator",
        "label": "Crop Stress Indicator",
        "short_label": "Stress",
        "placeholder": "49.5",
        "hint": "Composite indicator reflecting stress signals observed in the crop.",
        "step": "any",
        "min": "0",
        "max": "100",
    },
    {
        "name": "Pest_Damage",
        "label": "Pest Damage",
        "short_label": "Pest Damage",
        "placeholder": "49.5",
        "hint": "Estimated pest damage score or severity observed in the field.",
        "step": "any",
        "min": "0",
        "max": "100",
    },
]

FIELD_GROUPS = [
    {
        "title": "Terrain and Plant Signals",
        "description": "Core wheat growth indicators derived from canopy structure, vegetation indices, and plant condition.",
        "fields": FEATURE_FIELDS[:6],
    },
    {
        "title": "Weather and Water Conditions",
        "description": "Environmental conditions that shape crop stress, hydration, and short-term field performance.",
        "fields": [
            FEATURE_FIELDS[6],
            FEATURE_FIELDS[7],
            FEATURE_FIELDS[8],
            FEATURE_FIELDS[9],
            FEATURE_FIELDS[10],
            FEATURE_FIELDS[13],
        ],
    },
    {
        "title": "Soil and Pressure Indicators",
        "description": "Field health factors related to soil balance, weeds, pests, and broader crop pressure.",
        "fields": [
            FEATURE_FIELDS[11],
            FEATURE_FIELDS[12],
            FEATURE_FIELDS[14],
            FEATURE_FIELDS[15],
            FEATURE_FIELDS[16],
        ],
    },
]

DEMO_PRESET_TEMPLATES = [
    {
        "id": "healthy-cautious",
        "name": "Healthy / cautious",
        "values": {
            "Elevation_Data": "87.714053",
            "Canopy_Coverage": "12.437326",
            "NDVI": "0.410857",
            "SAVI": "0.643837",
            "Chlorophyll_Content": "0.713168",
            "Leaf_Area_Index": "1.397098",
            "Temperature": "29.425809",
            "Humidity": "52.304148",
            "Rainfall": "10.654384",
            "Wind_Speed": "2.606610",
            "Soil_Moisture": "13.419882",
            "Soil_pH": "6.561781",
            "Organic_Matter": "1.554045",
            "Water_Flow": "34.786117",
            "Weed_Coverage": "3.275139",
            "Crop_Stress_Indicator": "94.000000",
            "Pest_Damage": "17.000000",
        },
    },
    {
        "id": "unhealthy-alert",
        "name": "Unhealthy alert",
        "values": {
            "Elevation_Data": "91.506133",
            "Canopy_Coverage": "85.194725",
            "NDVI": "0.546845",
            "SAVI": "0.368530",
            "Chlorophyll_Content": "0.594742",
            "Leaf_Area_Index": "2.593671",
            "Temperature": "27.667904",
            "Humidity": "89.331283",
            "Rainfall": "9.227965",
            "Wind_Speed": "1.214457",
            "Soil_Moisture": "24.303800",
            "Soil_pH": "6.874632",
            "Organic_Matter": "11.363036",
            "Water_Flow": "44.246635",
            "Weed_Coverage": "2.865900",
            "Crop_Stress_Indicator": "95.000000",
            "Pest_Damage": "45.000000",
        },
    },
    {
        "id": "healthy-strong",
        "name": "Healthy / stronger",
        "values": {
            "Elevation_Data": "50.852443",
            "Canopy_Coverage": "55.461738",
            "NDVI": "0.571149",
            "SAVI": "0.655834",
            "Chlorophyll_Content": "2.479294",
            "Leaf_Area_Index": "2.566800",
            "Temperature": "26.619472",
            "Humidity": "53.864937",
            "Rainfall": "84.471636",
            "Wind_Speed": "0.612640",
            "Soil_Moisture": "30.829143",
            "Soil_pH": "6.725695",
            "Organic_Matter": "8.261021",
            "Water_Flow": "33.112274",
            "Weed_Coverage": "1.935799",
            "Crop_Stress_Indicator": "13.000000",
            "Pest_Damage": "20.000000",
        },
    },
]

FEATURE_NAMES = [field["name"] for field in FEATURE_FIELDS]
FEATURE_BY_NAME = {field["name"]: field for field in FEATURE_FIELDS}
REFERENCE_COMPARISON_FIELDS = [
    "Canopy_Coverage",
    "NDVI",
    "Chlorophyll_Content",
    "Soil_Moisture",
    "Crop_Stress_Indicator",
    "Pest_Damage",
]
REFERENCE_INSIGHT_FIELDS = [
    "Canopy_Coverage",
    "Leaf_Area_Index",
    "Soil_Moisture",
    "Soil_pH",
    "Crop_Stress_Indicator",
    "Pest_Damage",
]
PROJECT_VISUALS = [
    {
        "filename": "rf_feature_importance.png",
        "title": "Random Forest feature importance",
        "description": "Existing project visual showing which wheat variables mattered most during Random Forest training experiments.",
        "caption": "Source: code/drafts/rf_feature_importance.png",
    },
    {
        "filename": "scatter_wheat_stage4.png",
        "title": "Wheat stage scatter view",
        "description": "Existing project scatter plot used during exploratory work to inspect how wheat records cluster at a specific growth stage.",
        "caption": "Source: code/drafts/scatter_wheat_stage4.png",
    },
]

FEATURE_GROUPS = {
    "Soil": ["Soil_Moisture", "Soil_pH", "Organic_Matter"],
    "Weather": ["Temperature", "Humidity", "Rainfall", "Wind_Speed"],
    "Vegetation": ["Canopy_Coverage", "NDVI", "SAVI", "Chlorophyll_Content", "Leaf_Area_Index"],
    "Stress": ["Crop_Stress_Indicator", "Pest_Damage", "Weed_Coverage"],
    "Water": ["Soil_Moisture", "Rainfall", "Water_Flow"],
    "Field conditions": ["Elevation_Data", "Canopy_Coverage", "Weed_Coverage"],
}

STATIC_FEATURE_IMPORTANCE = {
    "Crop_Stress_Indicator": 0.12,
    "Pest_Damage": 0.11,
    "NDVI": 0.10,
    "Soil_Moisture": 0.09,
    "Chlorophyll_Content": 0.09,
    "Canopy_Coverage": 0.08,
    "Leaf_Area_Index": 0.07,
    "Humidity": 0.06,
    "Rainfall": 0.06,
    "Temperature": 0.05,
    "Soil_pH": 0.05,
    "Organic_Matter": 0.04,
    "Water_Flow": 0.03,
    "Weed_Coverage": 0.03,
    "Elevation_Data": 0.02,
    "SAVI": 0.02,
    "Wind_Speed": 0.02,
}

FEATURE_REASON_RULES = {
    "NDVI": {
        "healthy": "NDVI suggests stronger vegetation vigor than the unhealthy wheat profile.",
        "unhealthy": "NDVI sits closer to the weaker vegetation pattern seen in unhealthy wheat.",
    },
    "Canopy_Coverage": {
        "healthy": "Canopy coverage looks more like the denser healthy wheat profile.",
        "unhealthy": "Canopy coverage drifts away from the healthier wheat profile.",
    },
    "Chlorophyll_Content": {
        "healthy": "Chlorophyll content points to more active, healthier leaf conditions.",
        "unhealthy": "Chlorophyll content is lower than the healthier wheat reference.",
    },
    "Leaf_Area_Index": {
        "healthy": "Leaf area indicates stronger plant development.",
        "unhealthy": "Leaf area is less aligned with the healthy wheat pattern.",
    },
    "Soil_Moisture": {
        "healthy": "Soil moisture is supportive of crop hydration.",
        "unhealthy": "Soil moisture is not aligned with the healthy wheat reference band.",
    },
    "Soil_pH": {
        "healthy": "Soil pH sits closer to the healthy wheat balance.",
        "unhealthy": "Soil pH is away from the healthier soil balance.",
    },
    "Organic_Matter": {
        "healthy": "Organic matter looks closer to the healthier soil profile.",
        "unhealthy": "Organic matter is less aligned with the healthy soil profile.",
    },
    "Crop_Stress_Indicator": {
        "healthy": "The crop stress indicator is lower than the unhealthy reference.",
        "unhealthy": "Crop stress is elevated compared with the healthy wheat profile.",
    },
    "Pest_Damage": {
        "healthy": "Pest damage remains limited compared with unhealthy wheat cases.",
        "unhealthy": "Pest damage is contributing to the warning signal.",
    },
    "Weed_Coverage": {
        "healthy": "Weed coverage stays relatively controlled.",
        "unhealthy": "Weed coverage suggests more competition around the crop.",
    },
    "Temperature": {
        "healthy": "Temperature is closer to the healthy wheat reference.",
        "unhealthy": "Temperature sits away from the healthier reference band.",
    },
    "Humidity": {
        "healthy": "Humidity aligns better with the healthy wheat profile.",
        "unhealthy": "Humidity is closer to the unhealthy wheat pattern.",
    },
    "Rainfall": {
        "healthy": "Rainfall conditions look more supportive of healthy wheat.",
        "unhealthy": "Rainfall conditions are closer to the unhealthy reference.",
    },
    "Water_Flow": {
        "healthy": "Water flow is closer to the healthy field profile.",
        "unhealthy": "Water flow is less aligned with healthy wheat conditions.",
    },
    "Elevation_Data": {
        "healthy": "Field elevation sits closer to the healthy wheat reference.",
        "unhealthy": "Field elevation is nearer the unhealthy wheat reference pattern.",
    },
    "SAVI": {
        "healthy": "SAVI supports a healthier vegetation signal.",
        "unhealthy": "SAVI leans toward the weaker vegetation pattern.",
    },
    "Wind_Speed": {
        "healthy": "Wind conditions remain closer to the healthy wheat profile.",
        "unhealthy": "Wind conditions are less aligned with the healthy profile.",
    },
}


def find_existing_file(*candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def clamp(value, lower=0.0, upper=100.0):
    return max(lower, min(upper, value))


def normalize_against_bounds(value, lower, upper):
    if upper <= lower:
        return 50.0
    return clamp(((value - lower) / (upper - lower)) * 100.0)


def percentile_from_sorted(sorted_values, value):
    if sorted_values is None or len(sorted_values) == 0:
        return 0.0
    index = int(np.searchsorted(sorted_values, value, side="right"))
    return clamp((index / len(sorted_values)) * 100.0)


def summarize_range_position(value, stats):
    if value < stats["q1"]:
        return "Below the middle 50% of wheat records"
    if value > stats["q3"]:
        return "Above the middle 50% of wheat records"
    return "Inside the middle 50% of wheat records"


def classify_percentile(percentile):
    if percentile <= 25:
        return "low"
    if percentile >= 75:
        return "high"
    return "balanced"


def load_wheat_reference():
    dataset_path = find_existing_file(
        BASE_DIR / "src" / "agriculture_dataset.csv",
        DRAFTS_DIR / "agriculture_dataset.csv",
    )
    if dataset_path is None:
        return None

    usecols = set(["Crop_Type", "Crop_Health_Label", *FEATURE_NAMES])
    dataset = pd.read_csv(dataset_path, usecols=lambda column: column in usecols)
    wheat = dataset[dataset["Crop_Type"].astype(str).str.lower() == "wheat"].copy()
    reference_columns = [name for name in FEATURE_NAMES if name in wheat.columns]
    if "Crop_Health_Label" not in wheat.columns:
        return None

    numeric_reference = wheat[reference_columns + ["Crop_Health_Label"]].apply(pd.to_numeric, errors="coerce").dropna()
    if numeric_reference.empty:
        return None

    feature_stats = {}
    sorted_values = {}

    for field in FEATURE_FIELDS:
        name = field["name"]
        if name not in numeric_reference.columns:
            continue

        series = numeric_reference[name].astype(float)
        healthy_series = numeric_reference.loc[numeric_reference["Crop_Health_Label"] == 1, name].astype(float)
        unhealthy_series = numeric_reference.loc[numeric_reference["Crop_Health_Label"] == 0, name].astype(float)

        feature_stats[name] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q3": float(series.quantile(0.75)),
            "healthy_mean": float(healthy_series.mean()),
            "unhealthy_mean": float(unhealthy_series.mean()),
        }
        sorted_values[name] = np.sort(series.to_numpy(dtype=float))

    healthy_share = float(numeric_reference["Crop_Health_Label"].mean()) * 100.0
    unhealthy_share = 100.0 - healthy_share

    return {
        "dataset_path": str(dataset_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "sample_count": int(len(numeric_reference)),
        "healthy_share": round(healthy_share, 1),
        "unhealthy_share": round(unhealthy_share, 1),
        "feature_stats": feature_stats,
        "sorted_values": sorted_values,
    }


WHEAT_REFERENCE = load_wheat_reference()
MODEL_FEATURE_IMPORTANCE = get_feature_importance_map() or STATIC_FEATURE_IMPORTANCE


def format_display_number(value):
    if value is None:
        return ""
    numeric = float(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    if numeric.is_integer():
        return f"{numeric:.0f}"
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def build_home_highlights():
    reference_count = WHEAT_REFERENCE["sample_count"] if WHEAT_REFERENCE else 0
    class_count = get_model_class_count() or 2
    return [
        {"value": f"{len(FEATURE_FIELDS)}", "label": "model inputs used by the prediction form"},
        {"value": f"{reference_count:,}", "label": "wheat records available in the local reference dataset"},
        {"value": f"{class_count}", "label": "possible crop-health classes returned by the saved model"},
    ]


def build_home_snapshot():
    reference_count = WHEAT_REFERENCE["sample_count"] if WHEAT_REFERENCE else 0
    return {
        "input_count": len(FEATURE_FIELDS),
        "reference_count": f"{reference_count:,}",
        "class_count": get_model_class_count() or 2,
    }


def build_how_it_works():
    input_count = len(FEATURE_FIELDS)
    return [
        f"You enter {input_count} wheat-field measurements that match the exact inputs used by the trained model.",
        "The backend reuses the saved scaler and Random Forest classifier from the original project pipeline.",
        "The interface returns a healthy or unhealthy wheat prediction, a confidence score, and visual summaries built from your submitted values.",
    ]


def build_demo_presets():
    presets = []
    for preset in DEMO_PRESET_TEMPLATES:
        prediction_result = make_prediction(preset["values"])
        presets.append(
            {
                **preset,
                "outcome": prediction_result["prediction"],
                "confidence_text": f'{prediction_result["confidence"]}% confidence',
            }
        )
    return presets


def build_field_groups():
    groups = deepcopy(FIELD_GROUPS)
    for group in groups:
        for field in group["fields"]:
            stats = WHEAT_REFERENCE["feature_stats"].get(field["name"]) if WHEAT_REFERENCE else None
            field["placeholder"] = format_display_number(stats["median"]) if stats else "Enter value"
    return groups


HOME_HIGHLIGHTS = build_home_highlights()
HOME_SNAPSHOT = build_home_snapshot()
HOW_IT_WORKS = build_how_it_works()
DEMO_PRESETS = build_demo_presets()


def empty_form_values():
    return {field["name"]: "" for field in FEATURE_FIELDS}


def coerce_form_values(form_data):
    return {field["name"]: form_data.get(field["name"], "").strip() for field in FEATURE_FIELDS}


def build_alignment_scores(form_values):
    scores = {}
    if WHEAT_REFERENCE is None:
        return scores

    for feature_name in FEATURE_NAMES:
        value = form_values.get(feature_name, "")
        stats = WHEAT_REFERENCE["feature_stats"].get(feature_name)
        if value == "" or stats is None:
            continue

        numeric_value = float(value)
        distance_to_healthy = abs(numeric_value - stats["healthy_mean"])
        distance_to_unhealthy = abs(numeric_value - stats["unhealthy_mean"])
        total_distance = distance_to_healthy + distance_to_unhealthy
        if total_distance == 0:
            scores[feature_name] = 50.0
        else:
            scores[feature_name] = round(clamp((distance_to_unhealthy / total_distance) * 100.0), 2)

    return scores


def build_radar_payload(alignment_scores):
    labels = []
    values = []

    for group_name, feature_names in FEATURE_GROUPS.items():
        group_scores = [alignment_scores[name] for name in feature_names if name in alignment_scores]
        if not group_scores:
            continue
        labels.append(group_name)
        values.append(round(sum(group_scores) / len(group_scores), 2))

    return {
        "labels": labels,
        "values": values,
    }


def build_feature_importance_payload(alignment_scores):
    driver_rows = []

    for feature_name in FEATURE_NAMES:
        importance = MODEL_FEATURE_IMPORTANCE.get(feature_name, STATIC_FEATURE_IMPORTANCE.get(feature_name, 0.0))
        if importance <= 0:
            continue

        alignment_score = alignment_scores.get(feature_name, 50.0)
        impact_strength = importance * (abs(alignment_score - 50.0) / 50.0)
        driver_rows.append(
            {
                "name": feature_name,
                "label": FEATURE_BY_NAME[feature_name]["label"],
                "importance": float(importance),
                "alignment": alignment_score,
                "direction": "healthy" if alignment_score >= 50 else "unhealthy",
                "driver_score": impact_strength,
            }
        )

    ranked_rows = sorted(driver_rows, key=lambda item: item["driver_score"], reverse=True)
    top_rows = ranked_rows[:6]
    max_score = max((row["driver_score"] for row in top_rows), default=1.0)

    chart_rows = []
    for row in top_rows:
        normalized = round((row["driver_score"] / max_score) * 100.0, 2) if max_score else 0.0
        chart_rows.append({**row, "chart_score": normalized})

    return {
        "labels": [row["label"] for row in chart_rows],
        "values": [row["chart_score"] for row in chart_rows],
        "colors": ["rgba(45, 122, 76, 0.82)" if row["direction"] == "healthy" else "rgba(171, 79, 58, 0.82)" for row in chart_rows],
        "drivers": chart_rows,
    }


def build_why_prediction(prediction, confidence, driver_payload):
    driver_rows = driver_payload["drivers"][:3]
    reasons = []

    for row in driver_rows:
        templates = FEATURE_REASON_RULES.get(
            row["name"],
            {
                "healthy": f"{row['label']} is closer to the healthy wheat reference.",
                "unhealthy": f"{row['label']} is closer to the unhealthy wheat reference.",
            },
        )
        reasons.append(templates[row["direction"]])

    if prediction == "Healthy wheat crop":
        action = (
            "Maintain current field management and keep watching stress, pest damage, and moisture to preserve the healthy profile."
            if confidence >= 65
            else "The crop still lands on the healthy side, but with some uncertainty, so continue monitoring stress and pest indicators closely."
        )
    else:
        action = (
            "Inspect stress, pest pressure, and water-related conditions first, because those are the strongest warning signals in this profile."
            if confidence >= 60
            else "The crop trends unhealthy, but the margin is moderate, so validate the field with extra scouting before acting."
        )

    return {
        "title": "Why this prediction",
        "reasons": reasons,
        "action": action,
    }


def build_risk_context(prediction, confidence):
    if prediction == "Unhealthy wheat crop" and confidence >= 65:
        return {
            "level": "High risk",
            "theme": "unhealthy",
            "description": "The model is confidently flagging this wheat profile as unhealthy.",
        }
    if prediction == "Unhealthy wheat crop" or confidence < 60:
        return {
            "level": "Medium risk",
            "theme": "warning",
            "description": "This field needs closer attention because the result is either negative or borderline.",
        }
    return {
        "level": "Low risk",
        "theme": "healthy",
        "description": "The field profile aligns more comfortably with healthy wheat conditions.",
    }


def build_reference_overview():
    if WHEAT_REFERENCE is None:
        return None
    return {
        "sample_count": f"{WHEAT_REFERENCE['sample_count']:,}",
        "healthy_share": WHEAT_REFERENCE["healthy_share"],
        "unhealthy_share": WHEAT_REFERENCE["unhealthy_share"],
        "dataset_path": WHEAT_REFERENCE["dataset_path"],
    }


def build_dashboard_payload(probabilities, confidence, alignment_scores, driver_payload):
    return {
        "confidence": confidence,
        "uncertainty": round(max(0.0, 100.0 - confidence), 2),
        "probabilities": probabilities,
        "radar": build_radar_payload(alignment_scores),
        "importance": {
            "labels": driver_payload["labels"],
            "values": driver_payload["values"],
            "colors": driver_payload["colors"],
        },
    }


def build_result_context(prediction_result, form_values):
    prediction = prediction_result["prediction"]
    confidence = prediction_result["confidence"]
    probabilities = prediction_result["probabilities"]
    is_healthy = prediction == "Healthy wheat crop"
    confidence_band = (
        "High confidence" if confidence >= 85 else "Moderate confidence" if confidence >= 70 else "Low confidence"
    )

    explanation = (
        "The submitted profile looks closer to healthy wheat examples in the training data. Continue monitoring moisture, canopy strength, and pest pressure."
        if is_healthy
        else "The submitted profile is closer to unhealthy wheat examples in the training data. Check stress, pests, weeds, and water-related conditions first."
    )

    alignment_scores = build_alignment_scores(form_values)
    driver_payload = build_feature_importance_payload(alignment_scores)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "theme": "healthy" if is_healthy else "unhealthy",
        "heading": "Field conditions look favorable" if is_healthy else "Field conditions may need attention",
        "confidence_band": confidence_band,
        "explanation": explanation,
        "risk": build_risk_context(prediction, confidence),
        "why_prediction": build_why_prediction(prediction, confidence, driver_payload),
        "reference_overview": build_reference_overview(),
        "dashboard_payload": build_dashboard_payload(probabilities, confidence, alignment_scores, driver_payload),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        page_name="home",
        highlights=HOME_HIGHLIGHTS,
        how_it_works=HOW_IT_WORKS,
        home_snapshot=HOME_SNAPSHOT,
    )


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/project-visuals/<path:filename>")
def project_visual(filename):
    allowed = {visual["filename"] for visual in PROJECT_VISUALS}
    if filename not in allowed or not DRAFTS_DIR.exists():
        abort(404)
    return send_from_directory(DRAFTS_DIR, filename)


@app.route("/predict", methods=["GET", "POST"])
@app.route("/try", methods=["GET", "POST"])
def try_model():
    form_values = empty_form_values()
    result = None
    error = None

    if request.method == "POST":
        form_values = coerce_form_values(request.form)
        try:
            prediction_result = make_prediction(request.form)
            result = build_result_context(prediction_result, form_values)
        except (KeyError, TypeError, ValueError):
            error = "Please enter a valid numeric value for every required field."

    return render_template(
        "predict.html",
        page_name="predict",
        field_groups=build_field_groups(),
        form_values=form_values,
        demo_presets=DEMO_PRESETS,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )
