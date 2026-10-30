def generate_multimodal_score(
    clinical_probability: float,
    byra_steatosis_probability: float,
    behsof_nafld_probability: float
):
    """
    Generate a multimodal AI evidence score by combining
    clinical and ultrasound model outputs.

    NOTE:
    This is an AI evidence aggregation score.
    It is NOT a clinically validated disease probability.
    """

    # -------------------------------------------------
    # Model weights
    # -------------------------------------------------

    CLINICAL_WEIGHT = 0.70
    BYRA_WEIGHT = 0.10
    BEHSOF_WEIGHT = 0.20

    # -------------------------------------------------
    # Individual contributions
    # -------------------------------------------------

    clinical_contribution = (
        clinical_probability * CLINICAL_WEIGHT
    )

    byra_contribution = (
        byra_steatosis_probability * BYRA_WEIGHT
    )

    behsof_contribution = (
        behsof_nafld_probability * BEHSOF_WEIGHT
    )

    # -------------------------------------------------
    # Final multimodal score
    # -------------------------------------------------

    final_score = (
        clinical_contribution
        + byra_contribution
        + behsof_contribution
    )

    # -------------------------------------------------
    # Evidence levels
    # -------------------------------------------------

    def get_evidence_level(probability):
        if probability >= 0.70:
            return "High"
        elif probability >= 0.40:
            return "Moderate"
        else:
            return "Low"

    clinical_level = get_evidence_level(
        clinical_probability
    )

    byra_level = get_evidence_level(
        byra_steatosis_probability
    )

    behsof_level = get_evidence_level(
        behsof_nafld_probability
    )

    # -------------------------------------------------
    # Agreement
    # -------------------------------------------------

    levels = [
        clinical_level,
        byra_level,
        behsof_level
    ]

    high_count = levels.count("High")

    if high_count == 3:
        agreement = "Strong"
    elif high_count == 2:
        agreement = "Moderate"
    else:
        agreement = "Mixed"

    # -------------------------------------------------
    # Return result
    # -------------------------------------------------

    return {
        "score": round(final_score * 100, 1),

        "clinical": {
            "probability": round(
                clinical_probability * 100, 1
            ),
            "weight": CLINICAL_WEIGHT,
            "contribution": round(
                clinical_contribution * 100, 2
            ),
            "evidence_level": clinical_level
        },

        "byra": {
            "probability": round(
                byra_steatosis_probability * 100, 1
            ),
            "weight": BYRA_WEIGHT,
            "contribution": round(
                byra_contribution * 100, 2
            ),
            "evidence_level": byra_level
        },

        "behsof": {
            "probability": round(
                behsof_nafld_probability * 100, 1
            ),
            "weight": BEHSOF_WEIGHT,
            "contribution": round(
                behsof_contribution * 100, 2
            ),
            "evidence_level": behsof_level
        },

        "agreement": agreement
    }