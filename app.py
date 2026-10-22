from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numbers
import numpy as np
import shap

app = Flask(__name__)

# ==================================================
# 1. LOAD TRAINED MODEL AND FEATURE LIST
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "nutrition_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")

model = joblib.load(MODEL_PATH)
features = list(joblib.load(FEATURES_PATH))


# ==================================================
# 2. FORM FIELD → MODEL FEATURE
# ==================================================

FORM_TO_FEATURE = {
    "energy": "energy_100g",
    "fat": "fat_100g",
    "saturated_fat": "saturated-fat_100g",
    "carbohydrates": "carbohydrates_100g",
    "sugars": "sugars_100g",
    "sodium": "sodium_100g",
    "protein": "proteins_100g",
    "fiber": "fiber_100g",
    "cholesterol": "cholesterol_100g",
    "trans_fat": "trans-fat_100g",
    "calcium": "calcium_100g",
    "iron": "iron_100g",
    "additives": "additives_n"
}

# ==================================================
# SHAP EXPLAINER
# ==================================================

explainer = shap.TreeExplainer(model)


# ==================================================
# 3. NUTRIENTS THAT CAN BE CONVERTED
# ==================================================

# These are nutritional quantities.
# additives_n is a count and is NOT converted.

CONVERTIBLE_FEATURES = {
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "sodium_100g",
    "proteins_100g",
    "fiber_100g",
    "cholesterol_100g",
    "trans-fat_100g",
    "calcium_100g",
    "iron_100g"
}


# ==================================================
# 4. GRADE AND RISK MAPPINGS
# ==================================================

GRADE_MAPPING = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}

RISK_MAPPING = {
    "A": "Low",
    "B": "Low",
    "C": "Medium",
    "D": "High",
    "E": "High"
}


# ==================================================
# CHILD-FOCUSED WHO THRESHOLD CHECK (SEPARATE FROM ML)
# Based on WHO South-East Asia Region Nutrient Profile Model (2017).
# Values are assessed per 100 g; energy input is kJ and sodium is g.
# ==================================================

WHO_CATEGORIES = {
    "confectionery": {"label": "1. Confectionery", "criteria": {"fat": ("fat_100g", 8.0, "g"), "sugars": ("sugars_100g", 6.0, "g"), "energy": ("energy_100g", 230.0, "kcal")}},
    "fine_bakery": {"label": "2. Fine bakery wares", "criteria": {"fat": ("fat_100g", 8.0, "g"), "sugars": ("sugars_100g", 6.0, "g"), "sodium": ("sodium_100g", 0.25, "g"), "energy": ("energy_100g", 230.0, "kcal")}},
    "ordinary_bakery": {"label": "3. Bread and ordinary bakery wares", "criteria": {"fat": ("fat_100g", 8.0, "g"), "sugars": ("sugars_100g", 6.0, "g"), "sodium": ("sodium_100g", 0.25, "g")}},
    "cereals": {"label": "4. Cereals", "criteria": {"fat": ("fat_100g", 12.0, "g"), "sugars": ("sugars_100g", 9.0, "g"), "sodium": ("sodium_100g", 0.35, "g")}},
    "savoury_snacks": {"label": "5a. Ready-to-eat savouries", "criteria": {"fat": ("fat_100g", 8.0, "g"), "added_sugars": ("added_sugars", 0.0, "g"), "sodium": ("sodium_100g", 0.25, "g"), "energy": ("energy_100g", 230.0, "kcal")}},
    "processed_nuts": {"label": "5b. Processed nuts", "criteria": {"added_sugars": ("added_sugars", 0.0, "g"), "sodium": ("sodium_100g", 0.05, "g")}},
    "fish_snacks": {"label": "5c. Fish-based snacks", "criteria": {"sugars": ("sugars_100g", 6.0, "g"), "sodium": ("sodium_100g", 0.25, "g"), "energy": ("energy_100g", 230.0, "kcal")}},
    "fruit_veg_juice": {"label": "6a. Fruit and vegetable juices", "criteria": {"sugars": ("sugars_100g", 6.0, "g"), "added_sugars": ("added_sugars", 0.0, "g")}},
    "dairy_drinks": {"label": "6b. Milk and dairy-based drinks", "criteria": {"fat": ("fat_100g", 7.0, "g"), "added_sugars": ("added_sugars", 0.0, "g")}},
    "flavoured_drinks": {"label": "6c. Water-based flavoured drinks", "criteria": {"sugars": ("sugars_100g", 2.0, "g"), "sodium": ("sodium_100g", 0.30, "g")}},
    "tea_coffee": {"label": "6d. Coffee, tea and herbal infusions", "criteria": {"sugars": ("sugars_100g", 2.0, "g")}},
    "cereal_beverages": {"label": "6e. Cereal, grain and tree nut-based beverages", "criteria": {"sugars": ("sugars_100g", 6.0, "g"), "sodium": ("sodium_100g", 0.20, "g")}},
    "frozen_desserts": {"label": "7. Frozen dairy-based desserts and edible ices", "criteria": {"fat": ("fat_100g", 8.0, "g"), "sugars": ("sugars_100g", 12.0, "g"), "sodium": ("sodium_100g", 0.10, "g"), "energy": ("energy_100g", 230.0, "kcal")}},
    "curded_desserts": {"label": "8. Curded dairy-based desserts", "criteria": {"fat": ("fat_100g", 7.0, "g"), "sugars": ("sugars_100g", 6.0, "g"), "sodium": ("sodium_100g", 0.10, "g"), "energy": ("energy_100g", 230.0, "kcal")}},
    "cheese": {"label": "9. Cheese and analogues", "criteria": {"fat": ("fat_100g", 20.0, "g"), "added_sugars": ("added_sugars", 0.0, "g"), "sodium": ("sodium_100g", 0.60, "g")}},
    "composite_foods": {"label": "10. Composite foods (prepared foods)", "criteria": {"fat": ("fat_100g", 8.0, "g"), "saturated_fat": ("saturated-fat_100g", 3.5, "g"), "sugars": ("sugars_100g", 9.0, "g"), "sodium": ("sodium_100g", 0.35, "g")}},
    "fats_oils": {"label": "11. Fats, oils and fat emulsions", "criteria": {"saturated_fat": ("saturated-fat_100g", 35.0, "g"), "sodium": ("sodium_100g", 0.10, "g")}},
    "pasta_noodles": {"label": "12. Pasta, noodles and similar products", "criteria": {"fat": ("fat_100g", 3.0, "g"), "sodium": ("sodium_100g", 0.25, "g")}},
    "fresh_meat_fish": {"label": "13. Fresh/frozen meat, poultry, game, fish and seafood", "criteria": {"fat": ("fat_100g", 15.0, "g")}},
    "processed_meat": {"label": "14a. Processed meat, poultry and game", "criteria": {"fat": ("fat_100g", 8.0, "g"), "sodium": ("sodium_100g", 0.40, "g")}},
    "processed_fish": {"label": "14b. Processed fish and seafood", "criteria": {"fat": ("fat_100g", 8.0, "g"), "saturated_fat": ("saturated-fat_100g", 3.0, "g"), "sodium": ("sodium_100g", 0.40, "g")}},
    "fresh_produce": {"label": "15. Fresh/frozen fruits, vegetables and legumes", "criteria": {}},
    "processed_produce": {"label": "16. Processed fruits and vegetables", "criteria": {"added_sugars": ("added_sugars", 0.0, "g"), "sodium": ("sodium_100g", 0.40, "g")}},
    "soy_products": {"label": "17. Solid-form soybean products", "criteria": {"fat": ("fat_100g", 12.0, "g"), "sugars": ("sugars_100g", 5.0, "g"), "added_sugars": ("added_sugars", 0.0, "g"), "sodium": ("sodium_100g", 0.10, "g")}},
    "sauces": {"label": "18. Sauces, dips and dressings", "criteria": {"fat": ("fat_100g", 12.0, "g"), "sugars": ("sugars_100g", 10.0, "g"), "sodium": ("sodium_100g", 0.30, "g")}},
}

WHO_FIELD_LABELS = {
    "fat": "Total fat", "saturated_fat": "Saturated fat",
    "sugars": "Total sugars", "added_sugars": "Added sugars",
    "sodium": "Sodium", "energy": "Energy"
}

def assess_child_nutrition(category_key, nutrients_per_100g, added_sugars=None):
    """Apply category-specific WHO thresholds; does not modify ML/SHAP data."""
    category = WHO_CATEGORIES[category_key]
    exceeded, within, not_assessed = [], [], []
    for criterion, (source_key, limit, unit) in category["criteria"].items():
        if source_key == "added_sugars":
            value = added_sugars
        else:
            value = nutrients_per_100g.get(source_key)
            if criterion == "energy" and value is not None:
                value = float(value) / 4.184  # model/form energy is kJ; WHO threshold is kcal
        label = WHO_FIELD_LABELS[criterion]
        if value is None:
            not_assessed.append({"name": label, "threshold": limit, "unit": unit,
                                 "reason": "Required value not supplied"})
        elif float(value) > limit:
            exceeded.append({"name": label, "value": round(float(value), 4),
                             "threshold": limit, "unit": unit})
        else:
            within.append({"name": label, "value": round(float(value), 4),
                           "threshold": limit, "unit": unit})
    if not category["criteria"]:
        status = "Special category — no numeric thresholds assessed"
        summary = "The WHO model treats this category specially; this simple numeric check does not make a pass/fail determination."
    elif exceeded:
        status = "One or more thresholds exceeded"
        summary = "At least one applicable WHO nutrient-profile threshold was exceeded. Under this model, the product would not meet the nutrient criteria for marketing to children."
    elif not_assessed:
        status = "Incomplete check"
        summary = "No exceeded threshold was identified among assessed values, but one or more criteria could not be assessed because required label information was not supplied."
    else:
        status = "No applicable thresholds exceeded"
        summary = "No applicable numeric threshold was exceeded using the values entered. This is not a child-friendly certification or a medical assessment."
    notes = []
    if category_key == "sauces":
        notes.append("WHO category 18 has a special portion-size provision; this implementation uses a simplified per-100 g comparison and does not implement that provision.")
    if category_key == "pasta_noodles":
        notes.append("For products requiring reconstitution/preparation, use nutrient values for the product prepared according to package directions.")
    return {"category": category["label"], "status": status, "summary": summary,
            "exceeded": exceeded, "within": within, "not_assessed": not_assessed,
            "notes": notes}


# ==================================================
# 5. DISPLAY NAMES
# ==================================================

DISPLAY_NAMES = {
    "energy_100g": "Energy",
    "fat_100g": "Fat",
    "saturated-fat_100g": "Saturated Fat",
    "carbohydrates_100g": "Carbohydrates",
    "sugars_100g": "Sugars",
    "sodium_100g": "Sodium",
    "proteins_100g": "Protein",
    "fiber_100g": "Fiber",
    "cholesterol_100g": "Cholesterol",
    "trans-fat_100g": "Trans Fat",
    "calcium_100g": "Calcium",
    "iron_100g": "Iron",
    "additives_n": "Additives"
}


# ==================================================
# 6. UNIT INFORMATION
# ==================================================

UNITS = {
    "energy_100g": "kJ",
    "fat_100g": "g",
    "saturated-fat_100g": "g",
    "carbohydrates_100g": "g",
    "sugars_100g": "g",
    "sodium_100g": "g",
    "proteins_100g": "g",
    "fiber_100g": "g",
    "cholesterol_100g": "g",
    "trans-fat_100g": "g",
    "calcium_100g": "g",
    "iron_100g": "g",
    "additives_n": "count"
}


# ==================================================
# 7. CONVERT PREDICTION TO GRADE
# ==================================================

def convert_prediction_to_grade(prediction):

    if isinstance(prediction, numbers.Number):

        numeric_prediction = float(prediction)

        if not numeric_prediction.is_integer():
            raise ValueError(
                f"Model returned a non-integer grade: {prediction}"
            )

        numeric_prediction = int(numeric_prediction)

        if numeric_prediction not in GRADE_MAPPING:
            raise ValueError(
                f"Unexpected numeric grade returned: {prediction}"
            )

        return GRADE_MAPPING[numeric_prediction]

    grade = str(prediction).strip().upper()

    if grade in ["A", "B", "C", "D", "E"]:
        return grade

    try:

        numeric_prediction = float(grade)

        if numeric_prediction.is_integer():

            numeric_prediction = int(numeric_prediction)

            if numeric_prediction in GRADE_MAPPING:
                return GRADE_MAPPING[numeric_prediction]

    except ValueError:
        pass

    raise ValueError(
        f"Unexpected grade returned by model: {prediction}"
    )


# ==================================================
# 8. CONVERT PER-SERVING → PER-100G
# ==================================================

def convert_serving_to_100g(value, serving_size):

    if serving_size <= 0:
        raise ValueError(
            "Serving size must be greater than zero."
        )

    return (value / serving_size) * 100



# ==================================================
# 10. CREATE NUTRIENT DISPLAY DATA
# ==================================================

def create_nutrient_display(nutrients):

    nutrient_display = []

    for feature in features:

        value_per_100g = nutrients[feature]

        item = {
            "feature": feature,
            "name": DISPLAY_NAMES.get(
                feature,
                feature
            ),
            "value": round(
                value_per_100g,
                4
            ),
            "unit": UNITS.get(
                feature,
                ""
            )
        }

        nutrient_display.append(item)

    return nutrient_display



# ==================================================
# 11. HOME PAGE
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==================================================
# SHAP-BASED INDIVIDUAL PREDICTION EXPLANATION
# ==================================================

def explain_prediction(input_data, prediction, top_n=6):
    """
    Explain which features pushed the prediction toward
    or away from the predicted grade.

    SHAP values are class-specific contributions.
    They do not establish causation.
    """

    try:
        shap_values = explainer.shap_values(input_data)

        classes = list(model.classes_)
        class_index = classes.index(prediction)

        # Handle SHAP's older list-based output
        if isinstance(shap_values, list):
            class_values = np.asarray(
                shap_values[class_index]
            )
            contributions = class_values[0]

        else:
            values = np.asarray(shap_values)

            if values.ndim == 2:
                # One output per feature
                contributions = values[0]

            elif values.ndim == 3:
                # Common newer format:
                # (samples, features, classes)
                if (
                    values.shape[0] == 1
                    and values.shape[1] == len(features)
                    and values.shape[2] == len(classes)
                ):
                    contributions = values[0, :, class_index]

                # Alternative format:
                # (samples, classes, features)
                elif (
                    values.shape[0] == 1
                    and values.shape[1] == len(classes)
                    and values.shape[2] == len(features)
                ):
                    contributions = values[0, class_index, :]

                else:
                    raise ValueError(
                        "Unexpected SHAP output shape: "
                        f"{values.shape}"
                    )

            else:
                raise ValueError(
                    "Unsupported SHAP output dimensions: "
                    f"{values.ndim}"
                )

        if len(contributions) != len(features):
            raise ValueError(
                "SHAP contribution count does not match "
                "the model feature count."
            )

        # Rank by absolute contribution
        ranked = sorted(
            zip(features, contributions),
            key=lambda item: abs(float(item[1])),
            reverse=True
        )

        influences = []

        for feature, contribution in ranked[:top_n]:

            contribution = float(contribution)

            if contribution > 0:
                direction = "toward"
            elif contribution < 0:
                direction = "away from"
            else:
                direction = "neutral"

            if direction == "toward":
                message = (
                    f"This feature pushed the model output "
                    f"toward Grade "
                    f"{convert_prediction_to_grade(prediction)}."
                )
            elif direction == "away from":
                message = (
                    f"This feature pushed the model output "
                    f"away from Grade "
                    f"{convert_prediction_to_grade(prediction)}."
                )
            else:
                message = (
                    "This feature had approximately zero "
                    "SHAP contribution for this grade."
                )

            influences.append({
                "feature": feature,
                "nutrient": DISPLAY_NAMES.get(
                    feature, feature
                ),
                "contribution": round(
                    contribution, 6
                ),
                "direction": direction,
                "message": message
            })

        return influences

    except Exception:
        app.logger.exception(
            "SHAP explanation failed"
        )
        return []


# ==================================================
# 12. PREDICTION PAGE
# ==================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    if request.method == "GET":

        return render_template(
            "predict.html"
        )

    try:



        
        # Food details (age is no longer required)
        food_name = request.form.get("food_name", "").strip()
        child_category = request.form.get("child_category", "").strip()
        added_sugars_input = request.form.get("added_sugars", "").strip()
        if child_category not in WHO_CATEGORIES:
            raise ValueError("Please select a food category for the Child-Focused Nutrition Check.")
        added_sugars_per_input_basis = None
        if added_sugars_input:
            try:
                added_sugars_per_input_basis = float(added_sugars_input)
            except ValueError:
                raise ValueError("Added sugars must be a valid number.")
            if not np.isfinite(added_sugars_per_input_basis) or added_sugars_per_input_basis < 0:
                raise ValueError("Added sugars must be a non-negative number.")
# ------------------------------------------
        # INPUT MODE
        # ------------------------------------------

        input_mode = request.form.get(
            "input_mode",
            "per_100g"
        ).strip().lower()

        if input_mode not in [
            "per_100g",
            "per_serving"
        ]:

            raise ValueError(
                "Invalid nutrition input mode."
            )


        # ------------------------------------------
        # SERVING SIZE
        # ------------------------------------------

        serving_input = request.form.get(
            "serving_size",
            ""
        ).strip()

        serving_size = None

        if serving_input:

            try:

                serving_size = float(
                    serving_input
                )

            except ValueError:

                raise ValueError(
                    "Serving size must be a valid number."
                )

            if serving_size <= 0:

                raise ValueError(
                    "Serving size must be greater than zero."
                )

        if (
            input_mode == "per_serving"
            and serving_size is None
        ):

            raise ValueError(
                "Serving size is required when "
                "nutrition values are entered per serving."
            )


        # ------------------------------------------
        # READ NUTRIENT VALUES
        # ------------------------------------------

        nutrients = {}

        for form_name, feature_name in FORM_TO_FEATURE.items():

            value = request.form.get(
                form_name,
                ""
            ).strip()

            if value == "":

                raise ValueError(
                    f"Please enter a value for "
                    f"{form_name.replace('_', ' ').title()}."
                )

            try:

                numeric_value = float(value)

            except ValueError:

                raise ValueError(
                    f"{form_name.replace('_', ' ').title()} "
                    f"must be a valid number."
                )

            if not pd.notna(numeric_value):

                raise ValueError(
                    f"{form_name.replace('_', ' ').title()} "
                    f"must be a valid number."
                )

            if numeric_value < 0:

                raise ValueError(
                    f"{form_name.replace('_', ' ').title()} "
                    f"cannot be negative."
                )

            nutrients[feature_name] = numeric_value


        # ------------------------------------------
        # CONVERT PER SERVING → PER 100G
        # ------------------------------------------

        if input_mode == "per_serving":

            for feature in CONVERTIBLE_FEATURES:

                nutrients[feature] = (
                    convert_serving_to_100g(
                        nutrients[feature],
                        serving_size
                    )
                )

            # additives_n remains unchanged
            # because it represents a count.

        added_sugars_per_100g = added_sugars_per_input_basis
        if input_mode == "per_serving" and added_sugars_per_100g is not None:
            added_sugars_per_100g = convert_serving_to_100g(
                added_sugars_per_100g, serving_size
            )

        # ------------------------------------------
        # VALIDATE MODEL FEATURES
        # ------------------------------------------

        missing_features = [
            feature
            for feature in features
            if feature not in nutrients
        ]

        if missing_features:

            raise ValueError(
                "Missing model features: "
                + ", ".join(missing_features)
            )


        # ------------------------------------------
        # PREPARE MODEL INPUT
        # ------------------------------------------

        input_data = pd.DataFrame(
            [
                [
                    nutrients[feature]
                    for feature in features
                ]
            ],
            columns=features
        )


        # ------------------------------------------
        # RANDOM FOREST PREDICTION
        # ------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        print(
            "Raw prediction:",
            prediction
        )

        if hasattr(model, "classes_"):

            print(
                "Model classes:",
                model.classes_
            )


        # ------------------------------------------
        # GRADE
        # ------------------------------------------

        grade = convert_prediction_to_grade(
            prediction
        )


        # ------------------------------------------
        # RISK
        # ------------------------------------------

        risk_category = RISK_MAPPING[
            grade
        ]


        # ------------------------------------------
        # PREDICTION PROBABILITIES
        # ------------------------------------------

        probabilities = {}

        if hasattr(
            model,
            "predict_proba"
        ):

            probability_values = (
                model.predict_proba(
                    input_data
                )[0]
            )

            for class_value, probability in zip(
                model.classes_,
                probability_values
            ):

                class_grade = (
                    convert_prediction_to_grade(
                        class_value
                    )
                )

                probabilities[
                    class_grade
                ] = round(
                    float(probability) * 100,
                    2
                )


        # ------------------------------------------
        # NUTRIENT DISPLAY
        # ------------------------------------------

        nutrient_display = create_nutrient_display(nutrients)
        # ------------------------------------------
        # SHAP-BASED MODEL INFLUENCE
        # ------------------------------------------

        influences = explain_prediction(
        input_data,
        prediction,
        top_n=6
        )

        # Separate WHO category-based child-focused check. The model input,
        # Random Forest prediction and SHAP explanation above remain unchanged.
        child_check = assess_child_nutrition(
            child_category, nutrients, added_sugars_per_100g
        )

    # ------------------------------------------
        # OUTPUT
        # ------------------------------------------

        return render_template(
    "output.html",
    grade=grade,
    risk=risk_category,
    food_name=food_name or "Not provided",
    serving_size=serving_size,
    input_mode=input_mode,
    nutrients=nutrients,
    nutrient_display=nutrient_display,
    probabilities=probabilities,
    influences=influences,
    child_check=child_check,
    child_category_key=child_category,
    added_sugars=added_sugars_per_100g
)

    # ==================================================
    # VALIDATION ERRORS
    # ==================================================

    except (
        ValueError,
        TypeError
    ) as e:

        return render_template(
            "predict.html",
            error=str(e)
        ), 400


    # ==================================================
    # UNEXPECTED ERRORS
    # ==================================================

    except Exception as e:

        app.logger.exception(
            "Prediction failed"
        )

        return render_template(
            "predict.html",
            error=(
                "Unable to generate prediction: "
                + str(e)
            )
        ), 500


# ==================================================
# 13. RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
