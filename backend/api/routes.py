"""
REST API routes for liver disease prediction
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from pathlib import Path
import pandas as pd
import io
import tempfile
from PIL import Image
import os
from utils.multimodal_fusion import generate_multimodal_score
from models.schemas import (
    PatientDataRequest,
    CompletePredictionResponse,
    BulkPredictionResponse,
    BulkPredictionResult,
    ErrorResponse
)

from utils.predictor import predict_complete
from utils.ultrasound import analyze_ultrasound



router = APIRouter(prefix="/api/predict", tags=["predictions"])


# @router.post("/individual", response_model=CompletePredictionResponse)
# async def predict_individual(patient_data: PatientDataRequest):
#     """
#     Predict disease risk for a single patient.
    
#     Returns:
#         Complete prediction with supervised, unsupervised, and SHAP results
#     """
#     try:
#         # Convert Pydantic model to dict
#         patient_dict = patient_data.model_dump()
        
#         # Get prediction
#         result = predict_complete(patient_dict)
        
#         # Build response
#         response = CompletePredictionResponse(
#             success=True,
#             patient_id=patient_data.patientId,
#             supervised=result["supervised"],
#             unsupervised=result.get("unsupervised"),
#             shap=result.get("shap")
#         )
        
#         return response
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Prediction failed: {str(e)}"
#         )


@router.post("/individual", response_model=CompletePredictionResponse)
async def predict_individual(
    patient_data: str = Form(...),
    ultrasound: UploadFile = File(None)
):
    """
    Predict disease risk for a single patient.

    Accepts:
    - Clinical patient data as JSON string
    - Optional ultrasound image

    Clinical and ultrasound predictions are kept separate.
    """

    temp_image_path = None

    try:
        # -------------------------------------------------
        # 1. Parse clinical patient data
        # -------------------------------------------------

        try:
            patient_data_obj = PatientDataRequest.model_validate_json(
                patient_data
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid patient data JSON: {str(e)}"
            )

        patient_dict = patient_data_obj.model_dump()

        # -------------------------------------------------
        # 2. Existing clinical ML prediction
        # -------------------------------------------------

        result = predict_complete(patient_dict)

        # -------------------------------------------------
        # 3. Optional ultrasound analysis
        # -------------------------------------------------

        ultrasound_result = None

        if ultrasound is not None:

            allowed_types = {
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/webp"
            }

            if ultrasound.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unsupported ultrasound image format. "
                        "Please upload JPG, JPEG, PNG, or WEBP."
                    )
                )

            image_bytes = await ultrasound.read()
            
            MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

            if len(image_bytes) > MAX_IMAGE_SIZE:
                 raise HTTPException(
        status_code=400,
        detail="Image is too large. Maximum allowed size is 10 MB."
    )

            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded ultrasound image is empty."
                )
                
                
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image_format = image.format

                allowed_formats = {"JPEG", "PNG", "WEBP"}

                if image_format not in allowed_formats:
                    raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format: {image_format}. "
                "Please upload JPEG, PNG, or WEBP."
            )
        )

            except HTTPException:
                raise

            except Exception:
                raise HTTPException(
        status_code=400,
        detail="Could not read the uploaded image."
    )


            # Get extension
            suffix = Path(
                ultrasound.filename or ""
            ).suffix.lower()

            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                suffix = ".jpg"

            # Save temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(image_bytes)
                temp_image_path = temp_file.name

            # Run Byra + BEHSOF
            ultrasound_result = analyze_ultrasound(
                temp_image_path
            )
            
        # -------------------------------------------------
# 4. Generate multimodal AI evidence score
# -------------------------------------------------

        multimodal_score = None

        if ultrasound_result is not None:

            ultrasound_analysis = ultrasound_result["ultrasound_analysis"]

            multimodal_score = generate_multimodal_score(
        clinical_probability=result["supervised"]["risk_probability"],
        byra_steatosis_probability=(
            ultrasound_analysis["byra"]["steatosis_probability"]
        ),
        behsof_nafld_probability=(
            ultrasound_analysis["behsof"]["nafld_probability"]
        )
    )

        # -------------------------------------------------
        # 4. Build final response
        # -------------------------------------------------



        # response = CompletePredictionResponse(
        #     success=True,
        #     patient_id=patient_data_obj.patientId,

        #     # Clinical ML
        #     supervised=result["supervised"],
        #     unsupervised=result.get("unsupervised"),
        #     shap=result.get("shap"),

        #     # Ultrasound ML
        #     ultrasound_analysis=(
        #         ultrasound_result["ultrasound_analysis"]
        #         if ultrasound_result
        #         else None
        #     )
        # )

        # return response
        
        response = CompletePredictionResponse(
    success=True,
    patient_id=patient_data_obj.patientId,

    # Clinical ML
    supervised=result["supervised"],
    unsupervised=result.get("unsupervised"),
    shap=result.get("shap"),

    # Ultrasound ML
    ultrasound_analysis=(
        ultrasound_result["ultrasound_analysis"]
        if ultrasound_result
        else None
    ),

    # Multimodal fusion
    multimodal_score=multimodal_score
)
        return response

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        # Delete temporary uploaded image
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception:
                pass



@router.post("/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(file: UploadFile = File(...)):
    """
    Predict disease risk for multiple patients from CSV file.
    
    Expected CSV columns:
    - age, gender, totalBilirubin, directBilirubin, alkalinePhosphatase,
    - sgptAlt, sgotAst, totalProteins, albumin, agRatio
    - Optional: patientId, name
    
    Returns:
        Bulk prediction results with summary statistics
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Convert DataFrame to list of dictionaries
        patients = df.to_dict('records')
        
        # Process each patient
        results = []
        for idx, patient_row in enumerate(patients):
            try:
                # Map CSV columns to PatientData format
                patient_dict = {
                    "age": float(patient_row.get("age", 0)),
                    "gender": str(patient_row.get("gender", "")).strip(),
                    "totalBilirubin": float(patient_row.get("totalBilirubin", 0)),
                    "directBilirubin": float(patient_row.get("directBilirubin", 0)),
                    "alkalinePhosphatase": float(patient_row.get("alkalinePhosphatase", 0)),
                    "sgptAlt": float(patient_row.get("sgptAlt", 0)),
                    "sgotAst": float(patient_row.get("sgotAst", 0)),
                    "totalProteins": float(patient_row.get("totalProteins", 0)),
                    "albumin": float(patient_row.get("albumin", 0)),
                    "agRatio": float(patient_row.get("agRatio", 0)),
                }
                
                # Optional fields
                if "patientId" in patient_row:
                    patient_dict["patientId"] = str(patient_row["patientId"])
                if "name" in patient_row:
                    patient_dict["name"] = str(patient_row["name"])
                
                # Get prediction
                result = predict_complete(patient_dict)
                
                # Build result object
                bulk_result = BulkPredictionResult(
                    patient_id=patient_dict.get("patientId"),
                    name=patient_dict.get("name"),
                    # Include original patient data
                    age=patient_dict["age"],
                    gender=patient_dict["gender"],
                    totalBilirubin=patient_dict["totalBilirubin"],
                    directBilirubin=patient_dict["directBilirubin"],
                    alkalinePhosphatase=patient_dict["alkalinePhosphatase"],
                    sgptAlt=patient_dict["sgptAlt"],
                    sgotAst=patient_dict["sgotAst"],
                    totalProteins=patient_dict["totalProteins"],
                    albumin=patient_dict["albumin"],
                    agRatio=patient_dict["agRatio"],
                    # Prediction results
                    supervised=result["supervised"],
                    unsupervised=result.get("unsupervised"),
                    shap=result.get("shap")
                )
                
                results.append(bulk_result)
                
            except Exception as e:
                # Skip invalid rows
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate summary statistics
        if results:
            high_risk_count = sum(1 for r in results if r.supervised.risk_probability > 0.7)
            medium_risk_count = sum(1 for r in results if 0.3 <= r.supervised.risk_probability <= 0.7)
            low_risk_count = sum(1 for r in results if r.supervised.risk_probability < 0.3)
            avg_probability = sum(r.supervised.risk_probability for r in results) / len(results)
            
            summary = {
                "high_risk": high_risk_count,
                "medium_risk": medium_risk_count,
                "low_risk": low_risk_count,
                "average_probability": round(avg_probability, 3),
                "total_processed": len(results)
            }
        else:
            summary = {
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "average_probability": 0.0,
                "total_processed": 0
            }
        
        # Build response
        response = BulkPredictionResponse(
            success=True,
            total_patients=len(results),
            results=results,
            summary=summary
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Bulk prediction failed: {str(e)}"
        )
